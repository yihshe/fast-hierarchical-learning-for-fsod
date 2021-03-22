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

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup

import detectron2.utils.comm as comm
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)

from detectron2.solver import get_default_optimizer_params
import torch 
import time

from IPython import embed


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

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
    
    # def run_step(self):
    #     """
    #     Implement the standard training logic described above.
    #     """
    #     assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
    #     start = time.perf_counter()
    #     """
    #     If you want to do something with the data, you can wrap the dataloader.
    #     """
    #     data = next(self._data_loader_iter)
    #     data_time = time.perf_counter() - start
    #     """
    #     If you want to do something with the losses, you can wrap the model.
    #     """
    #     loss_dict = self.model(data)
    #     losses = sum(loss_dict.values())
    #     a = [p for p in self.model.roi_heads.box_predictor.parameters()]
    #     embed()
    #     grad = torch.autograd.grad(losses, self.model.roi_heads.box_predictor.parameters(), create_graph=True)
    #     embed()
    #     """
    #     If you need to accumulate gradients or do something similar, you can
    #     wrap the optimizer with your custom `zero_grad()` method.
    #     """
    #     self.optimizer.zero_grad()
    #     losses.backward()
    #     embed()
    #     self._write_metrics(loss_dict, data_time)

    #     """
    #     If you need gradient clipping/scaling or other processing, you can
    #     wrap the optimizer with your custom `step()` method. But it is
    #     suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    #     """
    #     self.optimizer.step()

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
