import itertools
import json
import os
import time
import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.config import global_cfg
from detectron2.engine.train_loop import HookBase
from detectron2.evaluation.testing import flatten_results_dict

from detectron2.engine.hooks import PeriodicWriter, PeriodicCheckpointer
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from IPython import embed

__all__ = ["EvalHookFsdet", "PeriodicWriterFsdet", "PeriodCheckpointerFsdet"]


class EvalHookFsdet(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, cfg):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            cfg: config
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self.cfg = cfg

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        if comm.is_main_process() and results:
            # save evaluation results in json
            is_final = self.trainer.iter + 1 >= self.trainer.max_iter
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, 'inference'), exist_ok=True)
            output_file = 'res_final.json' if is_final else \
                'iter_{:07d}.json'.format(self.trainer.iter)
            with PathManager.open(os.path.join(self.cfg.OUTPUT_DIR, 'inference',
                                               output_file), 'w') as fp:
                json.dump(results, fp)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func

class PeriodicWriterFsdet(PeriodicWriter):
    # now the loss_dict of the initial model will also be written in logs.

    # def __init__(self, writers, period=10):
    #     """
    #     Args:
    #         writers (list[EventWriter]): a list of EventWriter objects
    #         period (int):
    #     """
    #     super().__init__(writers, period)
    
    def after_step(self):
        if self.trainer.iter == 0 or (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
            ):
            for writer in self._writers:
                writer.write()

class PeriodCheckpointerFsdet(PeriodicCheckpointer):

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            self.checkpointer.save(
                "{}_{:07d}".format(self.file_prefix, iteration), **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                # pyre-fixme[58]: `>` is not supported for operand types `int` and
                #  `Optional[int]`.
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

        if self.max_iter is not None:
            # pyre-fixme[58]
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)