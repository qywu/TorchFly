import os
import sys
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from omegaconf import DictConfig
from colorlog import colorlog
from typing import Any, Dict

from ...common.logging_util import configure_logging
from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger("torchfly.training.logger")
Trainer = Any


@Callback.register("log_handler")
class LogHandler(Callback):
    """
    Callback that handles all Tensorboard logging.
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        if config.logging.log_dir is None:
            self.log_dir = "logs"
        else:
            self.log_dir = config.logging.log_dir

        os.makedirs(self.log_dir, exist_ok=True)

        self.avg_loss = None
        self.smooth_coef = 0.9

    @handle_event(Events.INITIALIZE, priority=100)
    def report_init_config(self, trainer: Trainer):
        logger.info(self.config.pretty())

    @handle_event(Events.TRAIN_BEGIN, priority=195)
    def setup_logging(self, trainer: Trainer):
        """
        Configure logging
        """
        if trainer.master:
            # Setup logger
            root = logging.getLogger()
            if hasattr(self.config.logging, "level"):
                root.setLevel(getattr(logging, self.config.logging.level))
            else:
                root.setLevel("INFO")
            # Setup formaters
            file_formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
            if self.config.logging.color:
                stream_formater = colorlog.ColoredFormatter(
                    "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
                    "[%(log_color)s%(levelname)s%(reset)s] - %(message)s"
                )
            else:
                stream_formater = file_formatter
            # Setup handlers
            if self.config.training.num_gpus_per_node > 1:
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(stream_formater)
                root.addHandler(stream_handler)
            # append the log
            file_handler = logging.FileHandler(os.path.join(self.log_dir, f"experiment.log"), mode='a')
            file_handler.setFormatter(file_formatter)
            root.addHandler(file_handler)

            # Choose to log
            self.log_in_seconds = False
            if self.config.logging.steps_interval <= 0:
                if (self.config.logging.seconds_interval is None) or (self.config.logging.seconds_interval < 0):
                    # default log_steps_interval
                    logger.warning("logging.steps_interval is not set. The default is set to 10!")
                    self.config.logging.steps_interval = 10
                else:
                    self.log_in_seconds = True

    # Setup timing
    @handle_event(Events.TRAIN_BEGIN, priority=155)
    def setup_timer(self, trainer: Trainer):
        if trainer.master:
            self.last_log_time = time.time()
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
        self.tensorboard = SummaryWriter(log_dir=os.getcwd(), purge_step=trainer.global_step_count)

    @handle_event(Events.EPOCH_BEGIN)
    def setup_epoch_timer(self, trainer: Trainer):
        if trainer.master:
            if trainer.no_epoch_training:
                logger.info(f"Training total num of steps: {trainer.total_num_update_steps}")
            else:
                logger.info("Epoch %d/%d", trainer.epochs_trained + 1, trainer.total_num_epochs)
                self.epoch_start_time = time.time()

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer: Trainer):
        if trainer.master:
            if self.resume_training:
                self.log(trainer, trainer.batch_results)
                self.resume_training = False
            elif self.log_in_seconds:
                current_time = time.time()
                iter_elapsed_time = current_time - self.last_log_time

                if iter_elapsed_time > self.config.logging.seconds_interval:
                    self.log(trainer, trainer.batch_results)
            else:
                if (trainer.global_step_count + 1) % self.config.logging.steps_interval == 0:
                    self.log(trainer, trainer.batch_results)

    @handle_event(Events.EPOCH_END, priority=-100)
    def on_epoch_end(self, trainer: Trainer):
        if trainer.master:
            if not trainer.no_epoch_training:
                epoch_elapsed_time = time.time() - self.epoch_start_time
                logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

    @handle_event(Events.TRAIN_END)
    def on_train_end(self, trainer: Trainer):
        if trainer.master:
            logging.shutdown()
            self.tensorboard.close()

            # Info the start
            logger.info("Training Finishes!")

    @handle_event(Events.VALIDATE_END)
    def show_metrics(self, trainer: Trainer):
        for metric_name, value in trainer.validate_metrics.items():
            metric_name = metric_name[0].upper() + metric_name[1:]
            if trainer.no_epoch_training:
                logger.info(
                    f"Steps {trainer.global_step_count}: Validation {metric_name} {value:4.4f}"
                )
            else:
                logger.info(f"Epoch {trainer.epochs_trained}: Validation {metric_name} {value:4.4f}")
            if isinstance(value, float):
                self.tensorboard.add_scalar("validate/" + metric_name, value, global_step=trainer.global_step_count)

    def log(self, trainer: Trainer, batch_results: Dict[str, torch.Tensor]):
        """
        Args:
            trainer: Trainer class
            batch_results: Dict
        """
        updated_steps = trainer.global_step_count // self.config.training.gradient_accumulation_steps

        if trainer.no_epoch_training:
            percent = 100. * updated_steps / trainer.total_num_update_steps
        else:
            percent = 100. * trainer.local_step_count / trainer.num_training_batches

        _loss = batch_results['loss'].item()

        if self.avg_loss:
            self.avg_loss = self.avg_loss * self.smooth_coef + _loss * (1- self.smooth_coef)
        else:
            self.avg_loss = _loss

        iter_elapsed_time = time.time() - self.last_log_time
        elapsed_steps = trainer.global_step_count - self.last_log_global_step 
                
        speed = elapsed_steps * self.config.training.batch_size * self.config.training.num_gpus_per_node / iter_elapsed_time

        if trainer.no_epoch_training:
            logger.info(
                f"Train Steps - {updated_steps:<10} - "
                f"[{percent:7.4f}%] - Speed: {speed:4.1f} - "
                f"Loss: {self.avg_loss:8.6f}"
            )
        else:
            logger.info(
                f"Train Epoch: [{trainer.epochs_trained + 1}/{self.config.training.total_num_epochs}]"
                f" [{percent:7.4f}%] - Loss: {self.avg_loss:8.6f}"
            )

        if self.tensorboard:
            self.tensorboard.add_scalar("train/loss", _loss, trainer.global_step_count + 1)
            self.tensorboard.add_scalar("train/learning_rate", get_lr(trainer.optimizer), trainer.global_step_count + 1)

        self.last_log_time = time.time()
        self.last_log_global_step = trainer.global_step_count

def get_lr(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']