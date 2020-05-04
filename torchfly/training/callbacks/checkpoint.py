from typing import Any, Dict
import os
import sys
import time
import torch
import logging
from omegaconf import DictConfig
from apex import amp

from ..checkpointer import Checkpointer
from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)

__all__ = ["Checkpoint"]
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


@Callback.register("checkpoint")
class Checkpoint(Callback):
    """
    Callback that handles Checkpointing
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.checkpoint_dir = self.config.training.checkpointing.directory
        if self.checkpoint_dir is None:
            self.checkpoint_dir = "Checkpoints"

        self.rank, _ = get_rank()
        self.fix_amp_bug = False

        if self.config.training.checkpointing.async_save is None:
            self.config.training.checkpointing.async_save = False

        # Initialize Checkpointer
        self.checkpointer = Checkpointer(
            sync_every_save=True,
            async_save=self.config.training.checkpointing.async_save,
            num_checkpoints_to_keep=self.config.training.checkpointing.num_checkpoints_to_keep,
            keep_checkpoint_every_num_seconds=(self.config.training.checkpointing.keep_checkpoint_every_num_seconds),
            storage_dir=self.checkpoint_dir
        )

        # checkpointed states contain two parts: model and training progress
        self.restored_states = None

        # setup the timer
        if self.rank == 0:
            # only rank 0 can save files
            self.last_save_time = time.time()

    @handle_event(Events.INITIALIZE, priority=199)
    def setup_checkpointer(self, trainer: Trainer):
        # Checkpoint in seconds or steps
        if self.config.training.checkpointing.steps_interval > 0:
            self.checkpoint_in_seconds = False
        else:
            if self.config.training.checkpointing.seconds_interval < 0:
                self.checkpoint_in_seconds = False
                # save for every epoch
                self.config.training.checkpointing.steps_interval = trainer.epoch_num_training_steps - 1
            else:
                self.checkpoint_in_seconds = True

        # Search for the latest checkpoint
        logger.info("Try to restore the latest checkpoint")
        if self.config.training.resume.resume:
            self.restored_states = self.checkpointer.restore_latest_checkpoint()
            if self.restored_states:
                self.checkpointer.load_state_dict(self.restored_states[1]["checkpointer_state_dict"])
        else:
            self.restored_states = None

    @handle_event(Events.TRAIN_BEGIN, priority=129)
    def fix_amp_in_train(self, trainer: Trainer):
        if self.restored_states:
            # make sure optimzier state is empty
            for optimizer in trainer.optimizers:
                optimizer.state = {}

    @handle_event(Events.BACKWARD_END, priority=100)
    def fix_amp_in_backward(self, trainer: Trainer):
        if self.restored_states:
            if not self.fix_amp_bug:
                # We load the optimizer's states here
                if self.config.training.resume.resume_optimizers:
                    for idx, optimizer in enumerate(trainer.optimizers):
                        try:
                            optimizer.load_state_dict(self.restored_states[1]["optimizers_state_dict"][idx])
                        except:
                            if self.rank == 0:
                                logger.warning(f"Cannot Load Optimizer {idx}'s State!")

                if self.rank == 0:
                    logger.warning(
                        "A Stupid Solution just to fix AMP bug! You only do it once everytime loading a checkpoint"
                    )
                self.fix_amp_bug = True
            self.restored_states = None

    # @handle_event(Events.TRAIN_BEGIN, priority=170)
    # def load_trainer_counts(self, trainer: Trainer):
    #     # Resume the training
    #     if self.restored_states:
    #         trainer.epochs_trained = self.restored_states[1]["epochs_trained"]
    #         trainer.global_step_count = self.restored_states[1]["global_step_count"]
    #         trainer.local_step_count = self.restored_states[1]["local_step_count"]

    @handle_event(Events.TRAIN_BEGIN, priority=170)
    def load_checkpoint(self, trainer: Trainer):
        # Load the model
        # Resume the training
        if self.restored_states and self.config.training.resume.resume:
            # Model State
            if self.config.training.resume.resume_model:
                trainer.set_model_state(self.restored_states[0])
            # Everything Else
            trainer.set_trainer_state(self.restored_states[1])

    @handle_event(Events.BATCH_END)
    def save_checkpoint(self, trainer: Trainer):
        # Checkpointing
        if self.rank == 0:
            if self.checkpoint_in_seconds:
                current_time = time.time()
                # the elapsed time is longer than the seconds
                if (current_time - self.last_save_time) > self.config.training.checkpointing.seconds_interval:
                    self._save_trainer_state(trainer)
                    self.last_save_time = current_time
            else:
                if (trainer.global_step_count + 1) % self.config.training.checkpointing.steps_interval == 0:
                    self._save_trainer_state(trainer)

    def _save_trainer_state(self, trainer: Trainer):
        trainer_state_dict = trainer.get_trainer_state()
        self.checkpointer.save_checkpoint(
            "iter_" + str(trainer.global_step_count), trainer.get_model_state(), trainer_state_dict
        )

    def state_dict(self):
        return self.checkpointer.state_dict()

    def load_state_dict(self, state_dict):
        self.checkpointer.load_state_dict(state_dict)