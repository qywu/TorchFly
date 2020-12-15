from typing import Any, Dict
import os
import sys
import time
import torch
import logging
from omegaconf import DictConfig

from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)

Trainer = Any


def get_rank():
    """
    We use environment variables to pass the rank info
    Returns:
        rank: rank in the multi-node system 
        local_rank: local rank on a node
    """
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        rank = 0
        local_rank = 0
        os.environ["RANK"] = str(0)
        os.environ["LOCAL_RANK"] = str(0)
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

    return rank, local_rank


@Callback.register("inference")
class Inference(Callback):
    """
    Callback that handles Checkpointing
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.rank, _ = get_rank()

        # setup the timer
        if self.rank == 0:
            # only rank 0 can do inference
            self.last_save_time = time.time()
            self.started = False

    @handle_event(Events.INITIALIZE, priority=199)
    def setup_inference(self, trainer: Trainer):
        # Checkpoint in seconds or steps
        if self.config.training.inference.steps_interval > 0:
            self.inference_in_seconds = False
        else:
            if (not hasattr(
                self.config.training.inference, "seconds_interval"
            )) or self.config.training.inference.seconds_interval < 0:
                self.inference_in_seconds = False
                # validate for every epoch
                self.config.training.inference.steps_interval = trainer.epoch_num_training_steps - 1
            else:
                self.inference_in_seconds = True

        # Inference steps interval
        if self.config.training.inference.after_num_steps is None:
            self.inference_after_num_steps = 0
        else:
            self.inference_after_num_steps = self.config.training.inference.after_num_steps

    @handle_event(Events.TRAIN_BEGIN)
    def on_train_begin(self, trainer):
        # Start validation at the begining
        if self.rank == 0 and not self.started:
            self._inference(trainer)
            self.started = True

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer: Trainer):
        # Check inference
        if self.rank == 0:
            if trainer.global_step_count > self.inference_after_num_steps:
                if self.inference_in_seconds:
                    current_time = time.time()
                    # the elapsed time is longer than the seconds
                    if (current_time - self.last_save_time) > self.config.training.inference.seconds_interval:
                        self._inference(trainer)
                        self.last_save_time = current_time
                else:
                    if (trainer.global_step_count + 1) % self.config.training.inference.steps_interval == 0:
                        self._inference(trainer)

    def _inference(self, trainer):
        if trainer.validation_dataloader is not None:
            trainer.model.eval()
            trainer.model.is_training = False
            # BEGIN
            trainer.callback_handler.fire_event(Events.VALIDATE_BEGIN)

            trainer.tmp_vars["validate_metrics"] = trainer.validate()

            trainer.callback_handler.fire_event(Events.VALIDATE_END)

            trainer.model.train()
            trainer.model.is_training = True

        if trainer.test_dataloader is not None:
            trainer.model.eval()
            trainer.model.is_training = False

            trainer.callback_handler.fire_event(Events.TEST_BEGIN)

            trainer.tmp_vars["test_metrics"] = trainer.test()

            trainer.callback_handler.fire_event(Events.TEST_END)

            trainer.model.train()
            trainer.model.is_training = True

    def state_dict(self):
        state_dict = {"started": self.started}
        return state_dict

    def load_state_dict(self, state_dict):
        self.started = state_dict["started"]