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

from fsdet.optimization import DetectionLossProblem, GradientMask, WeightPredictor, FeatureProjector
from fsdet.meta_optimization.task.build import TaskSampler, TaskDataset
from pytracking.libs.tensorlist import TensorList
from .meta_inference import MetaCGTrainer
# from .meta_weight import WeightPredictor

import copy
import time
import json
import numpy as np
from typing import Dict, List, Optional

from IPython import embed

# TODO dynamically sample the task, write metrics, and save lambda, verify the logic of model update
class MetaLearner(TrainerBase):
    
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
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
        # TODO rts the weights can be randini here
        self.checkpointer.resume_or_load(cfg.META_PARAMS.MODEL_WEIGHTS, resume = False)
        self.model = model

        self.pretrained_params = copy.deepcopy(self.model.state_dict())

        # TODO rts
        self.task_sampler = TaskSampler(cfg)
        self.task_dataset = TaskDataset(cfg, self.pretrained_params)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize the parameters to meta learn
        # NOTE this line is to initialize vector and scalar lambda
        self.hyper_params = {}
        self.hyper_params['lambda'] = self.init_lambda(cfg, self.pretrained_params)
        # torch.save(list(self.hyper_params['lambda'].cpu()), os.path.join(cfg.OUTPUT_DIR, 'lambda_params_init.pt'))
        
        for param in self.hyper_params.keys():
            self.hyper_params[param].requires_grad = True

        self.weight_predictor = WeightPredictor(cfg.META_PARAMS.WEIGHT_PREDICTOR.FEAT_SIZE, cfg.META_PARAMS.WEIGHT_PREDICTOR.BOX_DIM)
        self.weight_predictor.to(self.device)
        # self.weight_predictor = None

        # NOTE learn a feature projector to see if it helps
        # self.feature_projector = FeatureProjector(cfg.META_PARAMS.WEIGHT_PREDICTOR.FEAT_SIZE)
        # self.feature_projector.to(self.device)
        self.feature_projector = None

        self.optimizer = self.build_optimizer(cfg, self.hyper_params, self.weight_predictor, self.feature_projector)

        self.start_iter = 0
        self.max_iter = cfg.META_PARAMS.NUM_ITER

        self.cfg = cfg
        
        self.register_hooks(self.build_hooks())
    
    def init_lambda(self, cfg, params):
        if cfg.META_PARAMS.REGULARIZATION_TYPE == 'scalar':
            return torch.tensor(cfg.META_PARAMS.INIT_LOSS_REG).float().to(self.device)
        elif cfg.META_PARAMS.REGULARIZATION_TYPE == 'feature wise':
            return TensorList(self.lambda_matrix(params)).float().to(self.device)
        else:
            return torch.tensor(0).float().to(self.device)

    @classmethod
    # TODO modify the initialization of lambda here to make it feature wise
    def lambda_matrix(cls, params, 
                      param_names = ['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred'],
                      BASE_TAR_SIZE = 40):
        mats = list([])
        for idx, param_name in enumerate(param_names):
            for is_weight in [True, False]:
                mats.append(cls.single_lambda_matrix(params, param_name, is_weight))
        
        mats = [mat for mat in mats if mat is not None]
        return mats 
    
    @classmethod
    def single_lambda_matrix(cls, params, param_name, is_weight):
        if not is_weight and param_name+'.bias' not in params.keys():
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        if is_weight:
            mat = torch.rand(params[weight_name].shape[1])
        else:
            mat = torch.rand(1)

        torch.nn.init.normal_(mat, mean=0, std=0.1)
        return mat
        

    # All hooks here requires the after_step(), which is now wrapped in NewtonCG
    # TODO the periodic writer can be saved for the outer loop of SGD, but is no longer needed for the train() of CG
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        ret = [
            hooks.IterationTimer(),
            # hooks.LRScheduler(self.optimizer, self.scheduler)
        ]

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            # ret.append(hooks.PeriodicWriter(self.build_writers()))
            ret.append(PeriodicWriterFsdet(self.build_writers()))
        return ret

    # TODO separate the output dir  
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
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter 

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                # with torch.autograd.set_detect_anomaly(True):
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()

                    # TODO also save the state_dict of weight predictor module, using checkpointer function
                    # how to save all learned params in an unified framework, to be discussed 
                    if (self.iter+1) % 200 == 0:
                        # torch.save(list(self.hyper_params['lambda'].cpu()), os.path.join(self.cfg.OUTPUT_DIR, 'lambda_params_iter{}.pt'.format(self.iter)))
                        torch.save(self.weight_predictor.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, 'weight_predictor_iter{}.pt'.format(self.iter)))
                        # torch.save(self.feature_projector.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, 'feature_projector_iter{}.pt'.format(self.iter)))
                self.iter += 1
            except Exception:
                logger.exception("Exception during meta training:")
                raise
            finally:
                self.after_train()
                
        # if hasattr(self, "_last_eval_results") and comm.is_main_process():
        #     verify_results(self.cfg, self._last_eval_results)
        #     return self._last_eval_results
    

    def run_step(self):
        self.hyper_params['lambda'].requires_grad_(True)

        assert self.model.training, "Model was changed to eval mode!"
        start = time.perf_counter()
        # dynamically sample a batch of tasks
        tasks = self.build_task_batch()
        data_time = time.perf_counter() - start
        
        # only for the case when batch size is 1
        task = tasks[0]
        loss_dict = dict({})
        # execute the inner loop for CG, TBD: return the loss metrics
        # TODO here also pass the module to the trainer (with learnable params)
        meta_cg_trainer = MetaCGTrainer(self.cfg, self.model, task, self.hyper_params, self.weight_predictor, self.feature_projector)
        meta_cg_trainer.train()
        
        loss_dict['support'] = meta_cg_trainer.loss_dict
    
        # execute the outer loop for SGD updating
        data_loader = self.build_train_loader(self.cfg, dataset=task['data']['query'])
        proposals, box_features = self.extract_features(data_loader)
            
        # TODO the base_params is params of the pseudo base classes, and process for training should be simplified
        problem = DetectionLossProblem(proposals, box_features, 
                                       regularization = None,
                                       mask = task['setup']['masks'], 
                                       base_params = copy.deepcopy(task['setup']['base_params']), 
                                       reg = 0.0)
        losses = problem(self.model, meta_cg_trainer.optimizer.x)
        loss_dict['query'] = problem.loss_dict

        self.optimizer.zero_grad()
        losses.backward()
        # print('backed')
        # embed()
        self._write_metrics(loss_dict['query'], data_time)
        self.metrics_writer(loss_dict)

        self.optimizer.step()
        # print('stepped')
        # embed()

        # clear the unnecessary memory usage
        self.hyper_params['lambda'].detach_()
        del tasks, meta_cg_trainer, data_loader, proposals, box_features, problem, losses
        torch.cuda.empty_cache()

        

    def metrics_writer(self, loss_dict):
        # TODO for vector lambda, norm of the vector can be monitored to observe the magnitude of regularization
        os.makedirs(os.path.join(self.cfg.OUTPUT_DIR, "training_metrics"), exist_ok=True)
        if self.cfg.META_PARAMS.REGULARIZATION_TYPE == 'scalar':
            with open(os.path.join(self.cfg.OUTPUT_DIR, "training_metrics", "lambda.json"),"a") as fp:
                metrics = {'lambda': self.hyper_params['lambda'].detach().cpu().item(), 
                        'grad': self.hyper_params['lambda'].grad.detach().cpu().item()}
                metrics['iteration'] = self.iter
                json.dump(metrics, fp)
                fp.write('\n')
                fp.close()

        if loss_dict['support'] is not None:
            with open(os.path.join(self.cfg.OUTPUT_DIR, "training_metrics", "support_loss.json"),"a") as fp:
                metrics = {k: v.detach().cpu().item() for k, v in loss_dict['support'].items()}
                metrics['total_loss'] = sum(metrics.values())
                metrics['iteration'] = self.iter
                json.dump(metrics, fp)
                fp.write('\n')
                fp.close()

        with open(os.path.join(self.cfg.OUTPUT_DIR, "training_metrics", "query_loss.json"),"a") as fp:
            metrics = {k: v.detach().cpu().item() for k, v in loss_dict['query'].items()}
            metrics['total_loss'] = sum(metrics.values())
            metrics['iteration'] = self.iter
            json.dump(metrics, fp)
            fp.write('\n')
            fp.close()


    def extract_features(self, data_loader):
        for batch_idx, data in enumerate(data_loader):
            if batch_idx == 0:
                proposals, box_features = self.model.extract_features(data)
            else:
                proposals_batch, box_features_batch = self.model.extract_features(data)
                proposals = [*proposals, *proposals_batch]
                box_features = torch.cat((box_features, box_features_batch), dim=0)
        return proposals, box_features
        
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

    def build_task_batch(self):
        # Build an infinite stream of task sampler later for arbitrary number of iterations
        # TODO generate metadata first, then generate the task and then register with metadata
        meta_task_samples = self.task_sampler.generate_tasks()
        task_set_names = self.task_sampler.register_tasks(meta_task_samples)
        tasks = self.task_dataset.build_tasks(task_set_names)
        return tasks

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
    def build_optimizer(cls, cfg, hyper_params, weight_predictor = None, feature_projector = None):
        """
        Returns:
            DetectionNewtonCG optimizer:

        It now calls :func:`fsdet.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        # params = [hyper_params['lambda']] if cfg.META_PARAMS.REGULARIZATION_TYPE == 'scalar' else hyper_params['lambda']
        
        # if weight_predictor is not None:
        #     params = [*params, *[p for p in weight_predictor.parameters()]]

        # NOTE pass the params of weight predictor only
        params = [p for p in weight_predictor.parameters()]
        # params = [p for p in feature_projector.parameters()]
        
        # TODO add the multistep lr scheduler later on if needed
        return torch.optim.SGD(
            params, 
            lr = cfg.META_PARAMS.OUTER_OPTIM.BASE_LR,
            momentum = cfg.META_PARAMS.OUTER_OPTIM.MOMENTUM,
            nesterov = cfg.META_PARAMS.OUTER_OPTIM.NESTEROV,
            weight_decay = cfg.META_PARAMS.OUTER_OPTIM.WEIGHT_DECAY,
        )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`fsdet.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        # TODO decide finally whether we need a scheduler in trail test
        pass

    @classmethod
    def build_train_loader(cls, cfg, dataset = None):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, dataset = dataset, extract_features=True)


   