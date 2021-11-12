from typing import Any, Dict
import os
import sys
import time
import tqdm
import datetime
import collections
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from colorlog import colorlog
import atexit

from torchfly.flyconfig import FlyConfig
from torchfly.training.callbacks.events import Events
from torchfly.training.callbacks.callback import Callback, handle_event
import logging

logger = logging.getLogger("flylogger")
Trainer = Any


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook, Spyder or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


IN_NOTEBOOK = isnotebook()

if IN_NOTEBOOK:
    try:
        from IPython.display import clear_output, display, HTML
        import matplotlib.pyplot as plt
    except:
        logger.warn("Couldn't import ipywidgets properly, progress bar will use console behavior")
        IN_NOTEBOOK = False


@Callback.register("train_logger")
class TrainLogger(Callback):
    """
    Callback that handles all Tensorboard logging.
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.cumulative_time = 0.0

        # Log in seconds or steps
        if config.logging.steps_interval > 0:
            self.log_in_seconds = False
        else:
            if config.logging.seconds_interval < 0:
                raise NotImplementedError("config.logging.seconds_interval must be larger than 0")
            self.log_in_seconds = True

        # # Train in epochs or steps
        if config.total_num.epochs > 0:
            self.training_in_epoch = True
        else:
            if config.total_num.update_steps < 0:
                raise NotImplementedError("config.total_num.updated_steps must be larger than 0")
            self.training_in_epoch = False

        # correctly handles exception
        atexit.register(self.__del__)

    @handle_event(Events.TRAIN_BEGIN, priority=100)
    def report_training_config(self, trainer: Trainer):
        logger.info(OmegaConf.to_yaml(self.config))

    # Setup timing
    @handle_event(Events.TRAIN_BEGIN, priority=155)
    def setup_timer(self, trainer: Trainer):
        self.last_log_time = time.time()
        self.epoch_start_time = time.time()
        # Info the start
        logger.info("Training Starts!")

        if trainer.global_step_count > 0:
            self.resume_training = True
            logger.info("Resume the training!")
        else:
            self.resume_training = False

        self.last_log_global_step = 0

    @handle_event(Events.TRAIN_BEGIN, priority=140)
    def setup_tensorboard(self, trainer: Trainer):
        # Setup tensorboard
        log_dir = os.path.join(os.getcwd(), "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        self.tensorboard = SummaryWriter(log_dir=log_dir, purge_step=trainer.global_step_count)

        if self.training_in_epoch:
            logger.info(f"Total num of epochs of for training: {trainer.total_num_epochs}")
        logger.info(f"Total num of update steps of for training: {trainer.total_num_update_steps}")

    @handle_event(Events.EPOCH_BEGIN)
    def setup_epoch_timer(self, trainer: Trainer):
        if self.training_in_epoch:
            logger.info("Epoch %d/%d training starts!", trainer.epochs_trained + 1, trainer.total_num_epochs)
            self.epoch_start_time = time.time()
        else:
            logger.info(f"Epoch {trainer.epochs_trained + 1} training starts!")

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer: Trainer):
        if self.resume_training:
            self.log(trainer)
            self.resume_training = False
        elif self.log_in_seconds:
            current_time = time.time()
            iter_elapsed_time = current_time - self.last_log_time

            if iter_elapsed_time > self.config.logging.seconds_interval:
                self.log(trainer)
        else:
            if (trainer.global_step_count + 1) % self.config.logging.steps_interval == 0:
                self.log(trainer)

    @handle_event(Events.EPOCH_END, priority=100)
    def on_epoch_end(self, trainer: Trainer):
        epoch_elapsed_time = time.time() - self.epoch_start_time
        logger.info(f"Epoch {trainer.epochs_trained + 1} has finished!")
        logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

    @handle_event(Events.TRAIN_END)
    def on_train_end(self, trainer: Trainer):
        #logging.shutdown()
        # self.tensorboard.close()
        logger.info("Training Finishes!")

    def log(self, trainer: Trainer):
        """
        Args:
            trainer: Trainer class
            log_dict: Dict
        """
        iter_elapsed_time = time.time() - self.last_log_time
        elapsed_steps = trainer.global_step_count - self.last_log_global_step

        # items_per_second = elapsed_steps * self.config.batch_size * trainer.gradient_accumulation_batches * trainer.world_size / iter_elapsed_time
        self.cumulative_time += iter_elapsed_time

        log_string = (f"Epoch {trainer.epochs_trained + 1:2d} ")

        # only know number of steps, has no info about number of batchs in a epoch
        if not self.training_in_epoch:
            percent = 100. * trainer.global_step_count / trainer.total_num_update_steps + 1e-6
            log_string = f"Steps {trainer.global_step_count + 1:5d} [{percent:7.4f}%]"
            eta = str(datetime.timedelta(seconds=int(self.cumulative_time / percent * 100 - self.cumulative_time)))
            log_string += f" | ETA:{eta}"
        # has info about number of batchs in a epoch
        elif trainer.epoch_num_batches is not None:
            percent = 100. * trainer.local_step_count / (max(trainer.epoch_num_batches // trainer.gradient_accumulation_batches, 1)) + 1e-6
            log_string += f"Steps {trainer.global_step_count + 1:5d} [{percent:7.4f}%]"
            epoch_elapsed_time = time.time() - self.epoch_start_time
            eta = str(datetime.timedelta(seconds=int(epoch_elapsed_time / percent * 100 - epoch_elapsed_time)))
            log_string += f" | ETA:{eta}"
        # training forever
        else:
            log_string += f"Steps {trainer.global_step_count + 1:5d}"
            logger.warning("Not designed this way! `trainer.total_num_update_steps` not set")

        # log_string += f" | item/s {items_per_second:5.1f}"

        metrics = trainer.model.get_training_metrics()

        for metric_name, value in metrics.items():
            metric_name.replace("/", "_")
            if isinstance(value, str):
                log_string += f" | {metric_name} {value}"
                # tensorboard
                try:
                    value = float(value)
                    self.tensorboard.add_scalar("train/" + metric_name, value, global_step=trainer.global_step_count)
                except:
                    pass
            elif isinstance(value, float):
                log_string += f" | {metric_name} {value:6.4f}"
                self.tensorboard.add_scalar("train/" + metric_name, value, global_step=trainer.global_step_count)
            elif isinstance(value, tuple):
                log_string += f" | {metric_name} {value[0]}"
                self.tensorboard.add_scalar("train/" + metric_name, value[1], global_step=trainer.global_step_count)
            else:
                raise NotImplementedError("Cannot parse metric!")

        logger.info(log_string)

        # self.tensorboard.add_scalar("train/items_per_second", items_per_second, trainer.global_step_count)
        self.tensorboard.add_scalar("train/cumulative_time", self.cumulative_time, trainer.global_step_count)

        self.last_log_time = time.time()
        self.last_log_global_step = trainer.global_step_count

    def __del__(self):
        logging.shutdown()
        logger.handlers.clear()

        if hasattr(self, "tensorboard"):
            self.tensorboard.close()

    def state_dict(self):
        state_dict = {"cumulative_time": self.cumulative_time}
        return state_dict

    def load_state_dict(self, state_dict):
        self.cumulative_time = state_dict["cumulative_time"]