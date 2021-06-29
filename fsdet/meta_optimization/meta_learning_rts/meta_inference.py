"""
Define the trainer for inference
"""
import argparse
import logging
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.engine.hooks import EvalHookFsdet, PeriodicWriterFsdet
from fsdet.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from fsdet.modeling import build_model
from fsdet.data import build_detection_train_loader
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    # build_detection_train_loader,
)
from detectron2.engine import hooks, SimpleTrainer, TrainerBase
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.data.build import get_detection_dataset_dicts

from torch.utils.data.sampler import SequentialSampler

from fsdet.data import *
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)

from typing import Dict, List, Optional
import numpy as np
from fsdet.optimization import DetectionLossProblem, MetaDetectionNewtonCG
from fsdet.optimization import GradientMask

from IPython import embed
import copy

class MetaCGTrainer(TrainerBase):
    
    def __init__(self, cfg, model, task, hyper_params, weight_predictor = None, feature_projector = None):
        """
        Args:
            cfg (CfgNode):
        """
        # Assume these objects must be constructed in this order.
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        self.task = task
        # TODO rts change the structure of randinit params for RTS
        self.init_model(model, self.task['setup']['randinit_params'])
        self.model = model
        # TODO rts base_params is not needed here for regularization
        # self.base_params = self.task['setup']['base_params']
        # self.mask = self.task['setup']['masks']
        # TODO rts if it is RTS, all meta_data should be provided
        self.NOVEL_CLASSES = self.task['setup']['novel_classes']
        self.IDMAP = self.task['setup']['id_map']

        self.loss_reg = hyper_params['lambda']
        self.weight_predictor = weight_predictor
        self.feature_projector = feature_projector

        self.data_loader = self.build_train_loader(cfg, dataset=self.task['data']['support'])
        self.problem = None
        self.optimizer = None
        
        self.loss_dict = None

        # TODO these parameters can be shared among tasks, add the task specific parameters
        self.start_iter = 0
        self.warmup_iter = cfg.CG_PARAMS.NUM_NEWTON_ITER_WARMUP
        self.max_iter = cfg.CG_PARAMS.NUM_NEWTON_ITER_WARMUP + cfg.CG_PARAMS.NUM_NEWTON_ITER # 20
        self.num_cg_iter = cfg.CG_PARAMS.NUM_CG_ITER # 2
        if isinstance(self.num_cg_iter, int):
            assert self.num_cg_iter!=0, "Number of CG iteration is 0!"
            if self.max_iter is None:
                self.max_iter = 1
            self.num_cg_iter = [self.num_cg_iter]*self.max_iter
        self.max_iter = len(self.num_cg_iter)
        # assert self.max_iter!=0, "Number of Newton iteration is 0!"

        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # TODO check finally which hook is needed
        # self.register_hooks(self.build_hooks())


    @classmethod
    def init_model(cls, model, randinit_params):
        # TODO NOTE for training novel model there is no need to overwrite the weights here as all are randinit
        state_dict = model.roi_heads.box_predictor.state_dict()
        for i, layer_name in enumerate(list(state_dict.keys())):
            state_dict[layer_name] = randinit_params[i].detach()

        model.roi_heads.box_predictor.load_state_dict(state_dict)            

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
            self.checkpointer.resume_or_load(
                self.cfg.MODEL.WEIGHTS, resume=resume
            ).get("iteration", -1)
            + 1
        )
        if not resume:
            self.start_iter = 0


    # All hooks here requires the after_step(), which is now wrapped in NewtonCG
    # TODO the periodic writer can be saved for the outer loop of SGD, but is no longer needed for the train() of CG
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = (
            0  # save some memory and time for PreciseBN
        )

        ret = [
            hooks.IterationTimer(),
            # hooks.LRScheduler(self.optimizer, self.scheduler),
            # PreciseBN can be removed acutually as .ENABLED is False
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(EvalHookFsdet(
            cfg.TEST.EVAL_PERIOD, test_and_save_results, self.cfg))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            # ret.append(hooks.PeriodicWriter(self.build_writers()))
            ret.append(PeriodicWriterFsdet(self.build_writers()))
        return ret

    # TODO separate the output dir for 
    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    # TODO this function as the inner loop for CG
    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting CG training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter 

        with EventStorage(self.start_iter) as self.storage:
            # TODO modify the train loader to extract features for individual task
            
            # NOTE ground truth box and extract features only for novel classes (TBD), pass feat to small net
            # which will give an initial weights (20 classes), 1024 
            # NOTE index of the gt_classes are not the actual class id, there is a transformation
            # using thing dataset id to continuous id, and novel classes

            # proposals, box_features = self.extract_features()
            proposals, box_features, gt_box_features, gt_classes = self.extract_features(extract_gt_box_features=True)
            logger.info("Extracted features from frozen layers")

            novel_init_weights = None

            # NOTE currently the feature projector is only for test
            if self.feature_projector is not None:
                box_features = self.feature_projector(box_features)
                gt_box_features = self.feature_projector(gt_box_features)

            if self.weight_predictor is not None:
                novel_gt_box_features = self.get_novel_gt_box_features(gt_box_features, gt_classes) 
                novel_init_weights = self.weight_predictor(novel_gt_box_features)

            self.problem = DetectionLossProblem(proposals, box_features, 
                                                # TODO rts change the reg type to None or feature_wise_rts
                                                # regularization = self.cfg.META_PARAMS.REGULARIZATION_TYPE,
                                                # mask = self.mask, 
                                                # base_params = copy.deepcopy(self.base_params), 
                                                # reg = self.loss_reg
                                                )
            self.optimizer = self.build_optimizer(self.cfg, self.model, self.problem, 
                                                  novel_init_weights, self.IDMAP, self.NOVEL_CLASSES)

            try:
                self.before_train()

                self.warm_up()

                for self.iter in range(self.warmup_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()

                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
    
    def get_novel_gt_box_features(self, gt_box_features, gt_classes):
        novel_gt_box_features = torch.zeros(len(self.NOVEL_CLASSES), gt_box_features.shape[1]).to(self.device)
        for idx, novel_class in enumerate(self.NOVEL_CLASSES):
            novel_id = self.IDMAP[novel_class]
            novel_gt_box_features[idx] = torch.mean(gt_box_features[gt_classes==novel_id],dim=0)
        
        # return novel_gt_box_features.detach_()
        return novel_gt_box_features

    def warm_up(self):
        """Warm-up period to simulate the overfitting for learning regularizing factor"""
        self.loss_reg.detach_()

        for self.iter in range(self.start_iter, self.warmup_iter):
            # self.optimizer.x.requires_grad_(True)

            self.before_step()
            self.run_step()
            self.after_step()

            self.optimizer.x.detach_()

        self.loss_reg.requires_grad_(True)
        # self.optimizer.x.requires_grad_(True)

    def extract_features(self, extract_gt_box_features = False):
        if not extract_gt_box_features:
            for batch_idx, data in enumerate(self.data_loader):
                if batch_idx == 0:
                    proposals, box_features = self.model.extract_features(data)
                else:
                    proposals_batch, box_features_batch = self.model.extract_features(data)
                    proposals = [*proposals, *proposals_batch]
                    box_features = torch.cat((box_features, box_features_batch), dim=0)
            
            # TODO feature projection (the bbox feature should also be applied to weight predictor, faster convergence)
            return proposals, box_features

        else:
            for batch_idx, data in enumerate(self.data_loader):
                if batch_idx == 0:
                    proposals, box_features, gt_box_features, gt_classes = self.model.extract_features(data, extract_gt_box_features)
                else:
                    proposals_batch, box_features_batch, gt_box_features_batch, gt_classes_batch = self.model.extract_features(data, extract_gt_box_features)
                    proposals = [*proposals, *proposals_batch]
                    box_features = torch.cat((box_features, box_features_batch), dim=0)
                    gt_box_features = torch.cat((gt_box_features, gt_box_features_batch), dim=0)
                    gt_classes = [*gt_classes, *gt_classes_batch]
                gt_classes = torch.tensor(gt_classes)
            return proposals, box_features, gt_box_features, gt_classes
    
    def run_step(self):
        self.cg_iter = self.num_cg_iter[self.iter]
        self.loss_dict = self.optimizer.run_newton_iter(self.cg_iter)
        self.optimizer.hessian_reg *= self.optimizer.hessian_reg_factor

        # time needed to load the data is 0 since all features were extracted before training
        # data_time = 0
        # TODO only return the last metrics after train()
        # self._write_metrics(loss_dict, data_time)
        
    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    @classmethod
    def build_optimizer(cls, cfg, model, problem, novel_init_weights, IDMAP, NOVEL_CLASSES):
        """
        Returns:
            DetectionNewtonCG optimizer:

        It now calls :func:`fsdet.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        # TODO the bg weights also need to be passed to the model
        return MetaDetectionNewtonCG(problem = problem, model = model, augmentation= cfg.CG_PARAMS.AUGMENTATION,
                                     novel_init_weights = novel_init_weights, 
                                     IDMAP = IDMAP, 
                                     NOVEL_CLASSES = NOVEL_CLASSES)

    @classmethod
    def build_train_loader(cls, cfg, dataset = None):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, dataset = dataset, extract_features=True)
