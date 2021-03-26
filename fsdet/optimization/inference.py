"""
Define the trainer for inference
"""
# TODO change the config file for training arguments
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
from .optimization import DetectionLossProblem, DetectionNewtonCG

from IPython import embed

class CGTrainer(TrainerBase):
    
    def __init__(self, cfg):
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
        # TODO check if the model get updated correctly
        model = self.build_model(cfg)
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        model.train()
        self.model=model
        self.data_loader = self.build_train_loader(cfg)
        self.problem = None
        self.optimizer = None

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
        )
        
        self.start_iter = 0
        # TODO, add the num_cg_iter in configuration SOLVER_CG
        # initialize max_iter: int and num_cg_iter: list
        self.max_iter = cfg.CG_PARAMS.NUM_NEWTON_ITER
        self.num_cg_iter = cfg.CG_PARAMS.NUM_CG_ITER
        if isinstance(self.num_cg_iter, int):
            assert self.num_cg_iter!=0, "Number of CG iteration is 0!"
            if self.max_iter is None:
                self.max_iter = 1
            self.num_cg_iter = [self.num_cg_iter]*self.max_iter
        self.max_iter = len(self.num_cg_iter)
        assert self.max_iter!=0, "Number of Newton iteration is 0!"

        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        # TODO check the resume_or_load function
        self.start_iter = (
            self.checkpointer.resume_or_load(
                self.cfg.MODEL.WEIGHTS, resume=resume
            ).get("iteration", -1)
            + 1
        )

    # TODO all hooks here requires the after_step(), which is now wrapped in NewtonCG
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
            # TODO involves before_ and after_step function
            hooks.IterationTimer(),
            # hooks.LRScheduler(self.optimizer, self.scheduler),
            # TODO PreciseBN can be removed acutually as .ENABLED is False
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

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        # TODO rewrite the training function with NewtonCG.run(), similar workflow to detectron2 train()

        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter 

        with EventStorage(self.start_iter) as self.storage:
            proposals, box_features = self.extract_features()
            logger.info("Extracted features from frozen layers")
            
            self.problem = DetectionLossProblem(proposals, box_features)
            self.optimizer = self.build_optimizer(self.cfg, self.model, self.problem)

            try:
                self.before_train()
                self.optimizer.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()

                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                # the losses and residuals returned by optimizer can be saved in debug mode
                self.optimizer.after_train()
                self.after_train()
                
        # TODO also return the losses and residuals as in run()
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results
    
    def extract_features(self):
        for batch_idx, data in enumerate(self.data_loader):
            if batch_idx == 0:
                proposals, box_features = self.model.extract_features(data)
            else:
                proposals_batch, box_features_batch = self.model.extract_features(data)
                proposals = [*proposals, *proposals_batch]
                box_features = torch.cat((box_features, box_features_batch), dim=0)
        return proposals, box_features

    def run_step(self):
        self.cg_iter = self.num_cg_iter[self.iter]
        loss_dict = self.optimizer.run_newton_iter(self.cg_iter)
        self.optimizer.hessian_reg *= self.optimizer.hessian_reg_factor

        # time needed to load the data is 0 since all features were extracted before training
        data_time = 0
        self._write_metrics(loss_dict, data_time)

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
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`fsdet.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        if not cfg.MUTE_HEADER:
            logger = logging.getLogger(__name__)
            logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model, problem):
        """
        Returns:
            DetectionNewtonCG optimizer:

        It now calls :func:`fsdet.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        # TODO more args to be passed to the optimizer
        # return build_optimizer(cfg, model)
        return DetectionNewtonCG(problem, model, 
        debug= cfg.CG_PARAMS.DEBUG, 
        analyze=cfg.CG_PARAMS.ANALYZE_CONVERGENCE, 
        plotting=cfg.CG_PARAMS.PLOTTING)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`fsdet.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        pass

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        sampler = SequentialSampler(dataset)
        return build_detection_train_loader(cfg, dataset=dataset, sampler=sampler)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(
                evaluators
            ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        dataset_name
                    )
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results