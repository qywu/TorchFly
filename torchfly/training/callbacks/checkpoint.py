import os
import sys
import time
import torch
import logging
from omegaconf import DictConfig
from typing import Any, Dict

from ..checkpointer import Checkpointer
from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)

__all__ = ["Checkpoint"]
Trainer = Any


@Callback.register("checkpoint")
class Checkpoint(Callback):
    """
    Callback that handles all Tensorboard logging.
    """
    @handle_event(Events.INITIALIZE, priority=199)
    def setup_checkpointer(self, trainer: Trainer):
        self.fix_amp_bug = False
        
        if self.config.saving.save_dir is None:
            # Saving directory
            self.config.saving.save_dir = "Checkpoints"
        else:
            self.config.saving.save_dir = self.config.saving.save_dir

        # Initialize Checkpointer
        self.checkpointer = Checkpointer(
            sync_every_save=True,
            num_checkpoints_to_keep=self.config.saving.num_checkpoints_to_keep,
            keep_checkpoint_every_num_seconds=(self.config.saving.keep_checkpoint_every_num_seconds),
            storage_dir=self.config.saving.save_dir
        )

        trainer.checkpointer = self.checkpointer

    @handle_event(Events.INITIALIZE, priority=195)
    def search_checkpoint(self, trainer: Trainer):
        logger.info("Try to restore the latest checkpoint")
        if self.config.training.resume:
            self.states = self.checkpointer.restore_latest_checkpoint()
            if self.states:
                self.checkpointer.load_state_dict(self.states["checkpointer_states"])
        else:
            self.states = None

    @handle_event(Events.BACKWARD_END, priority=100)
    def fix_amp(self, trainer: Trainer):
        if self.states:
            if not self.fix_amp_bug:
                try:
                    trainer.optimizer.load_state_dict(self.states["optimizer_states"])
                except:
                    logger.warning("Cannot Load Optimizer States!")
                logger.warning("A Stupid Solution just to fix AMP bug! You only do it once everytime loading a checkpoint")
                self.fix_amp_bug = True
            self.states = None

    @handle_event(Events.TRAIN_BEGIN, priority=160)
    def setup_saving_variables(self, trainer: Trainer):
        try:
            num_batches = len(trainer.train_loader)
        except TypeError:
            # if cannot deceide the length of train_loader
            # save checkpoint every 30 mins
            self.config.saving.seconds_interval = 1800

        # Saving
        self.save_in_seconds = False
        # if nothign about saving interval is set
        if (self.config.saving.steps_interval is None or self.config.saving.steps_interval < 0
           ) and (self.config.saving.seconds_interval is None or self.config.saving.seconds_interval < 0):
            # then save for every epoch
            self.config.saving.steps_interval = num_batches - 1

        if self.config.saving.steps_interval < 0 and self.config.saving.seconds_interval > 0:
            self.save_in_seconds = True

    @handle_event(Events.TRAIN_BEGIN, priority=170)
    def load_trainer_variables(self, trainer: Trainer):
        # Resume the training
        if self.states is not None:
            trainer.load_trainer_variables(self.states)


    @handle_event(Events.TRAIN_BEGIN, priority=130)
    def load_states(self, trainer: Trainer):
        # Load the model
        # Resume the training
        if self.states is not None:
            trainer.load_state_dict(self.states)
            # a temp solution to fix amp bug
            trainer.optimizer.state = {}
            logger.info(f"Loaded the saved checkpoint {self.states['file_path']}")
        else:
            logger.info("Not loading any checkpoint. Training from scratch!")

        # setup the timer
        if trainer.master:
            self.last_save_time = time.time()

    @handle_event(Events.BATCH_END)
    def save_checkpoint(self, trainer: Trainer):
        # Checkpointing
        if trainer.master:
            if self.save_in_seconds:
                current_time = time.time()
                # the elapsed time is longer than the seconds
                if (current_time - self.last_save_time) > self.config.saving.seconds_interval:
                    self.__save(trainer)
                    self.last_save_time = current_time
            else:
                if (trainer.global_step_count + 1) % self.config.saving.steps_interval == 0:
                    self.__save(trainer)

    def __save(self, trainer: Trainer):
        states = trainer.state_dict()
        states["checkpointer_states"] = self.checkpointer.state_dict()
        self.checkpointer.save_checkpoint("iter_" + str(trainer.global_step_count), states)
