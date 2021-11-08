"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import numpy as np
import torch

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup

import detectron2.utils.comm as comm
import json
import logging
import os
import time
from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks, launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)

from fsdet.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from fsdet.data import build_detection_test_loader_with_gt

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
            return PascalVOCDetectionEvaluator(dataset_name, output_folder)
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
    def test(cls, cfg, model, evaluators=None, with_gt = False, results_path = None):
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
            # TODO modify the mapper of test loader to make it also attach the instance for each image
            if with_gt is True:
                data_loader = build_detection_test_loader_with_gt(cfg, dataset_name)
            else:
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
            results_i = inference_on_dataset(model, data_loader, evaluator, with_gt = with_gt, results_path=results_path)
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




class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Trainer.build_model(cfg)
        self.check_pointer = DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR
        )

        self.best_res = None
        self.best_file = None
        self.all_res = {}

    def test(self, ckpt):
        self.check_pointer._load_model(self.check_pointer._load_file(ckpt))
        print("evaluating checkpoint {}".format(ckpt))
        res = Trainer.test(self.cfg, self.model)

        if comm.is_main_process():
            verify_results(self.cfg, res)
            print(res)
            if (self.best_res is None) or (
                self.best_res is not None
                and self.best_res["bbox"]["AP"] < res["bbox"]["AP"]
            ):
                self.best_res = res
                self.best_file = ckpt
            print("best results from checkpoint {}".format(self.best_file))
            print(self.best_res)
            self.all_res["best_file"] = self.best_file
            self.all_res["best_res"] = self.best_res
            self.all_res[ckpt] = res
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(self.cfg.OUTPUT_DIR, "inference", "all_res.json"),
                "w",
            ) as fp:
                json.dump(self.all_res, fp)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
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
        if args.eval_iter != -1:
            # load checkpoint at specified iteration
            ckpt_file = os.path.join(
                cfg.OUTPUT_DIR, "model_{:07d}.pth".format(args.eval_iter - 1)
            )
            resume = False
        else:
            # load checkpoint at last iteration
            ckpt_file = cfg.MODEL.WEIGHTS
            resume = True
            # ckpt_file = os.path.join(
            #     cfg.OUTPUT_DIR, "model_init.pth"
            # )
            # resume = False
            
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            ckpt_file, resume=resume
        )
        # TODO if test with saved results, for the novel setting, then also save to a separate file maybe
        # start from the inner function
        if args.test_with_gt:
            res = Trainer.test(cfg, model, with_gt = args.test_with_gt)
        elif args.results_path is not None:
            res = Trainer.test(cfg, model, results_path = args.results_path)
        else:
            res = Trainer.test(cfg, model)

        setting = "_new_setting" if args.results_path is not None else ""

        if comm.is_main_process():
            verify_results(cfg, res)
            # save evaluation results in json
            os.makedirs(
                os.path.join(cfg.OUTPUT_DIR, "inference", exist_ok=True)
            )
            with open(
                os.path.join(cfg.OUTPUT_DIR, "inference", "res_final{}.json".format(setting)),
                "w",
            ) as fp:
                json.dump(res, fp)
        return res
    elif args.eval_all:
        tester = Tester(cfg)
        all_ckpts = sorted(tester.check_pointer.get_all_checkpoint_files())
        for i, ckpt in enumerate(all_ckpts):
            ckpt_iter = ckpt.split("model_")[-1].split(".pth")[0]
            if ckpt_iter.isnumeric() and int(ckpt_iter) + 1 < args.start_iter:
                # skip evaluation of checkpoints before start iteration
                continue
            if args.end_iter != -1:
                if (
                    not ckpt_iter.isnumeric()
                    or int(ckpt_iter) + 1 > args.end_iter
                ):
                    # skip evaluation of checkpoints after end iteration
                    break
            tester.test(ckpt)
        return tester.best_res
    elif args.eval_during_train:
        tester = Tester(cfg)
        saved_checkpoint = None
        while True:
            if tester.check_pointer.has_checkpoint():
                current_ckpt = tester.check_pointer.get_checkpoint_file()
                if (
                    saved_checkpoint is None
                    or current_ckpt != saved_checkpoint
                ):
                    saved_checkpoint = current_ckpt
                    tester.test(current_ckpt)
            time.sleep(10)
    else:
        if comm.is_main_process():
            print(
                "Please specify --eval-only, --eval-all, or --eval-during-train"
            )


if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    parser = default_argument_parser()
    parser.add_argument(
        "--test-with-gt", action = "store_true", help="test with ground truth labels",
    )
    parser.add_argument(
        "--results-path", type=str, default = None, help="path of the saved results to evaluate",
    )
    args = parser.parse_args()
    if args.eval_during_train or args.eval_all:
        args.dist_url = "tcp://127.0.0.1:{:05d}".format(
            np.random.choice(np.arange(0, 65534))
        )
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
