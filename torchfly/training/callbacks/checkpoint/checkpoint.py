from typing import Any, Dict, Union
import os
import sys
import time
import torch
import logging
from omegaconf import DictConfig

from .checkpointer import Checkpointer
from ..events import Events
from ..callback import Callback, handle_event
from ....distributed import get_rank

logger = logging.getLogger("checkpointer")

__all__ = ["Checkpoint"]
Trainer = Any


@Callback.register("checkpoint")
class Checkpoint(Callback):
    """
    Callback that handles Checkpointing
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        # Check distributed
        if get_rank() != 0:
            raise NotImplementedError("Checkpoint callback can only be called for rank 0!")
        self.checkpointer = None
        self.storage_dir = None

    @handle_event(Events.TRAIN_BEGIN, priority=199)
    def setup_checkpointer(self, trainer: Trainer):
        # Initialize Checkpointer
        self.storage_dir = os.path.join(f"{trainer.trainer_name}_{trainer.stage_name}", trainer.config.checkpointing.directory)
        self.checkpointer = Checkpointer(
            sync_every_save=True,
            async_save=trainer.config.checkpointing.async_save,
            num_checkpoints_to_keep=trainer.config.checkpointing.num_checkpoints_to_keep,
            keep_checkpoint_every_num_seconds=(trainer.config.checkpointing.keep_checkpoint_every_num_seconds),
            storage_dir=self.storage_dir)

        # Checkpoint in epochs or steps
        if trainer.config.checkpointing.steps_interval < 0 and trainer.config.checkpointing.seconds_interval < 0:
            self.checkpoint_in_epoch = True
        else:
            self.checkpoint_in_epoch = False

        # Checkpoint in seconds or steps
        if trainer.config.checkpointing.steps_interval > 0 and trainer.config.checkpointing.seconds_interval > 0:
            raise ValueError(
                "Either `checkpointing.steps_interval` or `checkpointing.seconds_interval` can be set greater than 0!")
        elif trainer.config.checkpointing.steps_interval < 0 and trainer.config.checkpointing.seconds_interval > 0:
            self.checkpoint_in_seconds = True
        elif trainer.config.checkpointing.steps_interval > 0 and trainer.config.checkpointing.seconds_interval < 0:
            self.checkpoint_in_seconds = False
        else:
            self.checkpoint_in_seconds = False
        self.last_save_time = time.time()

    @handle_event(Events.BATCH_END)
    def save_checkpoint(self, trainer: Trainer):
        # Checkpointing
        if not self.checkpoint_in_epoch:
            if self.checkpoint_in_seconds:
                current_time = time.time()
                # the elapsed time is longer than the seconds
                if (current_time - self.last_save_time) > trainer.config.checkpointing.seconds_interval:
                    self._save_trainer_state(trainer)
                    self.last_save_time = current_time
            else:
                if (trainer.global_step_count + 1) % trainer.config.checkpointing.steps_interval == 0:
                    self._save_trainer_state(trainer)

    @handle_event(Events.EPOCH_END)
    def save_checkpoint_epoch(self, trainer: Trainer):
        # Checkpointing
        if self.checkpoint_in_epoch:
            self._save_trainer_state(trainer)

    @handle_event(Events.TRAIN_END)
    def save_checkpoint_stage(self, trainer: Trainer):
        trainer_state_dict = trainer.get_trainer_state()
        self.checkpointer.save_checkpoint("stage_end", trainer.get_model_state(),
                                          trainer_state_dict)

    def _save_trainer_state(self, trainer: Trainer):

        trainer_state_dict = trainer.get_trainer_state()
        self.checkpointer.save_checkpoint("iter_" + str(trainer.global_step_count), trainer.get_model_state(),
                                          trainer_state_dict)
        logger.info(f"Saved Checkpoint for Epoch {trainer.epochs_trained + 1} Iteration {trainer.global_step_count}!")

    def state_dict(self):
        return self.checkpointer.state_dict()

    def load_state_dict(self, state_dict):
        self.checkpointer.load_state_dict(state_dict)