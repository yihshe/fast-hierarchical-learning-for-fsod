"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

from fsdet.engine.hooks import EvalHookFsdet, PeriodicWriterFsdet
from fvcore.nn.precise_bn import get_bn_modules
from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup

import detectron2.utils.comm as comm
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks, launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)
from fsdet.data import build_detection_train_loader, ExtractedDataset

from detectron2.solver import get_default_optimizer_params
import torch 
import time

from IPython import embed
from detectron2.utils.events import EventStorage
from torch.nn.parallel.distributed import DistributedDataParallel
import logging
from fsdet.optimization.gradient_mask import GradientMask
from pytracking.libs.tensorlist import TensorList

import random

# The train_net.py has been modified according to this project. 
# To use the original one, please download a separate file.


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        self.data_loader_before = None
        self.data_loader_after = None
        self.dataset = None
        
        ## define variables used for masking out gradients for pretrained weights
        data_source = cfg.DATASETS.TRAIN[0].split('_')[0]
        base_model = torch.load(cfg.MODEL.PRETRAINED_BASE_MODEL)
        # mask_generator = GradientMask(data_source)
        # self.mask = mask_generator.create_mask(self.model.state_dict(), base_model['model'])

        self.proposals = None
        self.box_features = None
    
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
            hooks.LRScheduler(self.optimizer, self.scheduler),
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
            
        # annotate this function so that the final model will not be evaluated
        # def test_and_save_results():
        #     self._last_eval_results = self.test(self.cfg, self.model)
        #     return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(EvalHookFsdet(
        #     cfg.TEST.EVAL_PERIOD, test_and_save_results, self.cfg))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            # ret.append(hooks.PeriodicWriter(self.build_writers()))
            ret.append(PeriodicWriterFsdet(self.build_writers()))
        return ret

    def train(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.debug("Starting training from iteration {}".format(self.start_iter))
        print("Starting training from iteration {}".format(self.start_iter))
        # TODO check here and print the logger, finish at least part of the introduction today. table and figure needed. send to check.
        self.iter = self.start_iter

        with EventStorage(self.start_iter) as self.storage:
            self.data_loader_before = self.build_train_loader_feature(self.cfg, extract_features=True)
            
            start = time.perf_counter()

            proposals, box_features = self.extract_features(self.model, self.data_loader_before)
            
            extract_time = time.perf_counter()-start
            logger.info("Extracted features from frozen layers. Time needed: {}".format(extract_time))
            print("Extracted features from frozen layers. Time needed: {}".format(extract_time))
            # logger.info("Extracted features from frozen layers")
            
            # torch.save(proposals, 'checkpoints_temp/proposals_coco.pt')
            # torch.save(box_features, 'checkpoints_temp/box_features_coco.pt')
            # proposals = torch.load('checkpoints_temp/proposals_coco.pt')
            # box_features = torch.load('checkpoints_temp/box_features_coco.pt')
            # proposals = torch.load('checkpoints_temp/proposals_coco_novel_3lvbg.pt')
            # box_features = torch.load('checkpoints_temp/box_features_coco_novel_3lvbg.pt')

            # self.proposals = proposals
            # self.box_features = box_features

            self.dataset = ExtractedDataset(proposals, box_features)
            self.data_loader_after = self.build_train_loader_feature(self.cfg, dataset=self.dataset)
            self._data_loader_after_iter = iter(self.data_loader_after)

            try:
                self.before_train()
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
                self.after_train()
    
    @classmethod
    def extract_features(cls, model, data_loader):
        for batch_idx, data in enumerate(data_loader):
            if batch_idx == 0:
                proposals, box_features = model.extract_features(data)
            else:
                proposals_batch, box_features_batch = model.extract_features(data)
                proposals = [*proposals, *proposals_batch]
                box_features = torch.cat((box_features, box_features_batch), dim=0)
        return proposals, box_features

    # TODO rebuild the data loader to first extract features and proposals
    @classmethod
    def build_train_loader_feature(cls, cfg, extract_features = False, dataset = None):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if extract_features: 
            return build_detection_train_loader(cfg, extract_features=extract_features)
        else:
            assert dataset is not None, "Extracted dataset must be specified!"
            return build_detection_train_loader(cfg, dataset=dataset)

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
    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_after_iter)
        proposals, box_features = self.unzip_data(data)
        data_time = time.perf_counter() - start
        # proposals, box_features = self.proposals, self.box_features
        # data_time = 0

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model.losses_from_features(box_features, proposals)
        # loss_dict = self.model.losses_from_features(self.box_features, self.proposals)
        losses = sum(loss_dict.values())
        # print('loss update: {}'.format(loss_dict))

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        # self.mask_out_gradient()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def unzip_data(self, data):
        """
        Restore proposals and box_features from the data batch
        """
        proposals, box_features = zip(*data)
        box_features = torch.cat(box_features, dim=0)

        return proposals, box_features

    # def mask_out_gradient(self):
    #     for idx, p in enumerate(self.model.roi_heads.box_predictor.parameters()):
    #         p.grad *= self.mask[idx]
        

    # # Rewrite some classmethods for the implementation of LBFGS. Write a new subclass if needed.
    # @classmethod
    # def build_optimizer(cls, cfg, model):
    #     """
    #     Build an LBFGS optimizer from config.
    #     """
    #     # LBFGS does not support per-parameter options and parameter groups
    #     params = model.roi_heads.box_predictor.parameters()
    #     # params = model.parameters()
    #     return torch.optim.LBFGS(
    #         params,
    #         lr = cfg.SOLVER.BASE_LR,
    #         history_size = 1,
    #     )

    # def run_step(self):
    #     """
    #     Implement the training logic for LBFGS optimizer.
    #     """
    #     assert self.model.training, "[Trainer] model was changed to eval mode!"
    #     start = time.perf_counter()
    #     """
    #     If you want to do something with the data, you can wrap the dataloader.
    #     """
    #     data = next(self._data_loader_iter)
    #     data_time = time.perf_counter() - start

    #     """
    #     Calculate the losses again for monitoring
    #     """
    #     loss_dict = self.model(data)
    #     self._write_metrics(loss_dict, data_time)

    #     def closure():
    #         """
    #         If you want to do something with the losses, you can wrap the model.
    #         """
    #         loss_dict = self.model(data)
    #         losses = sum(loss_dict.values())

    #         """
    #         If you need to accumulate gradients or do something similar, you can
    #         wrap the optimizer with your custom `zero_grad()` method.
    #         """
    #         self.optimizer.zero_grad()
    #         losses.backward()

    #         return losses

    #     """
    #     If you need gradient clipping/scaling or other processing, you can
    #     wrap the optimizer with your custom `step()` method. But it is
    #     suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    #     """
    #     self.optimizer.step(closure)

    #     print('box_head', self.model.roi_heads.box_head.fc1.weight.grad)
    #     print('box_predictor', self.model.roi_heads.box_predictor.cls_score.weight.grad.abs().sum())

    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # TBD: modify the OUTPUT_DIR in cfg according to args
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # random.seed(10)

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
