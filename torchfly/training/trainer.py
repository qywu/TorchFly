import os
import sys
import ray
import math
import atexit
import signal
import time
import datetime
import logging
from typing import Any, List, Dict, Iterator, Callable
from omegaconf import DictConfig
import apex
from apex import amp
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import torch.multiprocessing as multiprocessing

from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.training.callbacks import TrainHandler, LogHandler, GradientClipNorm, Checkpoint, PlasmaHandler
from torchfly.training.checkpointer import Checkpointer
from torchfly.training.optimization import ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule, \
    WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule
from torchfly.common import move_to_device, configure_logging

logger = logging.getLogger(__name__)


def cycle_wrapper(dataloader):
    while True:
        for element in dataloader.__iter__():
            yield element


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module = None,
        train_loader: Iterator = None,
        train_loader_fn: Callable = None,
        validation_loader: Iterator = None,
        test_loader: Iterator = None,
    ):
        """
        Do not send anything to cuda in __init__
        """
        self.config = config

        # # Data Loading
        self.train_loader = train_loader
        self.train_loader_fn = train_loader_fn
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.model = model

        # local variables
        self.global_step_count = 0
        self.epochs_trained = 0
        self.local_step_count = 0
        self.no_epoch_training = False
        self.num_training_batches = 0
        self.device = None
        self.optimizer = None
        self.scheduler = None

        # constants
        self.total_num_update_steps = int(self.config.training.total_num_update_steps)
        self.total_num_steps = self.total_num_update_steps * int(self.config.training.gradient_accumulation_steps)
        self.total_num_epochs = int(self.config.training.total_num_epochs)

        self.callback_handler = CallbackHandler(
            config, trainer=self, callbacks=self.default_callbacks(), verbose=config.logging.level == "DEBUG"
        )
        atexit.register(self.__del__)

    def default_callbacks(self) -> List:
        callbacks = []
        callbacks.append(TrainHandler(self.config))
        callbacks.append(LogHandler(self.config))
        callbacks.append(GradientClipNorm(self.config))
        callbacks.append(Checkpoint(self.config))
        if self.config.training.plasma.initialize_plasma:
            callbacks.append(PlasmaHandler(self.config))
        return callbacks

    def train(self) -> Dict[str, Any]:
        self.callback_handler.fire_event(Events.INITIALIZE)

        if self.config.training.num_gpus_per_node > 1:
            logger.info("Initializing Distributed Training")

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(random.randint(20000, 29000))  # use a random port, but might collide
            os.environ["WORLD_SIZE"] = str(self.config.training.num_gpus_per_node)

            torch.multiprocessing.set_start_method('spawn')
            # multiprocessing.log_to_stderr()
            # TODO: Use Process instead of spawn
            self._spawn_context = torch.multiprocessing.spawn(
                self._train, args=(), nprocs=self.config.training.num_gpus_per_node, join=False, daemon=True
            )
            self._spawn_context.join()
        elif self.config.training.num_gpus_per_node == 1:
            logger.info("Initializing Single GPU Training")
            results = self._train(rank=0)
        else:
            raise NotImplementedError("Do you mean CPU training?")

        return results

    def _train(self, rank=0):
        self.rank = rank
        self.config.rank = rank
        self.master = rank == 0

        # Training Begin
        self.callback_handler.fire_event(Events.TRAIN_BEGIN)

        self.train_loader = cycle_wrapper(self.train_loader)

        for _ in range(self.global_step_count, self.total_num_steps):
            if self.local_step_count == 0:
                self.callback_handler.fire_event(Events.EPOCH_BEGIN)

            # The state should be perserved by torch.get_rng_state
            # However, this solution is not deterministic, but at least it ensures
            # the randomness when loading data
            self.batch = next(self.train_loader)

            # callback handler has access to trainer.batch
            self.callback_handler.fire_event(Events.BATCH_BEGIN)

            self.batch = move_to_device(self.batch, self.device)
            self.batch_results = self.train_iter(self.batch)

            # Update the model
            if (self.global_step_count + 1) % self.config.training.optimization.gradient_accumulation_steps == 0:
                self.callback_handler.fire_event(Events.STEP_BEGIN)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.callback_handler.fire_event(Events.STEP_END)

            self.callback_handler.fire_event(Events.BATCH_END)

            # Validation
            if self.master:
                if (self.global_step_count + 1) % self.config.training.validation_steps_interval == 0:
                    if self.validation_loader:
                        self.callback_handler.fire_event(Events.VALIDATE_BEGIN)
                        self.model.eval()
                        self.validate_metrics = self.validate()
                        self.callback_handler.fire_event(Events.VALIDATE_END)
                        self.model.train()

            if not self.no_epoch_training and (self.local_step_count + 1) % self.num_training_batches == 0:
                self.callback_handler.fire_event(Events.EPOCH_END)
                self.epochs_trained += 1
                self.local_step_count = 0
            else:
                self.local_step_count += 1

            # Post
            self.global_step_count += 1

        self.callback_handler.fire_event(Events.TRAIN_END)
        return {}

    def train_iter(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        results = self.model(batch)
        loss = results["loss"]

        if self.config.training.optimization.gradient_accumulation_steps > 1:
            loss = loss / self.config.training.optimization.gradient_accumulation_steps

        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
        if self.config.training.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.callback_handler.fire_event(Events.BACKWARD_END)
        return results

    def validate(self):
        self.model.eval()
        for batch in self.validation_loader:
            # send to cuda device
            batch = move_to_device(batch, self.device)
            with torch.no_grad():
                if self.config.training.num_gpus_per_node > 1:
                    self.model.module.predict(batch)
                else:
                    self.model.predict(batch)
        # get metrics
        if self.config.training.num_gpus_per_node > 1:
            metrics = self.model.module.get_metrics(reset=True)
        else:
            metrics = self.model.get_metrics(reset=True)
        return metrics

    def get_optimizer_parameters(self):
        """
        This function is used to set parameters with different weight decays
        """
        # default groups
        no_decay = ["bias", "Norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.optimization.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizer(self) -> Optimizer:
        optimizer_grouped_parameters = self.get_optimizer_parameters()
        lr = self.config.training.optimization.learning_rate
        optimizer_name = self.config.training.optimization.optimizer
        max_gradient_norm = self.config.training.optimization.max_gradient_norm
        betas = self.config.training.optimization.betas if self.config.training.optimization.betas else (0.9, 0.999)

        if optimizer_name == "AdamW":
            return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=betas)
        elif optimizer_name == "FusedAdam":
            return apex.optimizers.FusedAdam(optimizer_grouped_parameters, lr=lr, betas=betas)
        elif optimizer_name == "Adadelta":
            return torch.optim.Adadelta(optimizer_grouped_parameters, lr=lr)
        elif optimizer_name == "FusedLAMB":
            if max_gradient_norm < 0:
                max_gradient_norm = 1.0
            else:
                # avoid a second clip_grad_norm
                self.config.training.optimization.max_gradient_norm = -1
            return apex.optimizers.FusedLAMB(
                optimizer_grouped_parameters, lr=lr, betas=betas, max_grad_norm=max_gradient_norm
            )
        else:
            raise NotImplementedError

    def configure_scheduler(self) -> LambdaLR:
        scheduler_name = self.config.training.optimization.scheduler
        warmup_steps = self.config.training.optimization.warmup.warmup_steps
        warmup_cycle = self.config.traing.optimization.warmup.warmup_cosine_cycle

        if scheduler_name == "Constant":
            return ConstantLRSchedule(self.optimizer)
        elif scheduler_name == "WarmupConstant":
            return WarmupConstantSchedule(self.optimizer, warmup_steps)
        elif scheduler_name == "WarmupLinear":
            return WarmupLinearSchedule(self.optimizer, warmup_steps, self.total_num_update_steps)
        elif scheduler_name == "WarmupCosine":
            if warmup_cycle is None:
                warmup_cycle = 0.5
            return WarmupCosineSchedule(self.optimizer, warmup_steps, self.total_num_update_steps, cycles=warmup_cycle)
        elif scheduler_name == "WarmupCosineWithHardRestartsSchedule":
            if warmup_cycle is None:
                warmup_cycle = 0.5

            return WarmupCosineWithHardRestartsSchedule(
                self.optimizer, warmup_steps, self.total_num_update_steps, cycles=warmup_cycle
            )
        else:
            logger.error("Write your own version of `configure_scheduler`!")
            raise NotImplementedError

    def state_dict(self):
        trainer_state_dict = {
            "epoch": self.epochs_trained,
            "global_update_count": self.global_step_count,
            "local_update_count": self.local_step_count,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        # save amp states
        if self.config.training.optimization.fp16:
            trainer_state_dict["amp_state_dict"] = amp.state_dict()
        return trainer_state_dict

    def load_trainer_counts(self, trainer_state_dict: Dict[str, Any]):
        self.epochs_trained = trainer_state_dict["epoch"]
        self.global_step_count = trainer_state_dict["global_update_count"]
        self.local_step_count = trainer_state_dict["local_update_count"]

    def model_state_dict(self):
        if "DistributedDataParallel" in str(type(self.model)):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()

    def load_model_state_dict(self, model_state_dict: Dict[str, Any]):
        try:
            if "DistributedDataParallel" in str(type(self.model)):
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)
        except:
            logger.warning("Cannot restore model state!")

    def load_state_dict(self, trainer_state_dict: Dict[str, Any]):
        self.epochs_trained = trainer_state_dict["epoch"]
        self.global_step_count = trainer_state_dict["global_update_count"]
        self.local_step_count = trainer_state_dict["local_update_count"]

        try:
            self.scheduler.load_state_dict(trainer_state_dict["scheduler_state_dict"])
        except:
            logger.warning("Cannot restore scheduler state!")

        try:
            # cpu random state
            torch.set_rng_state(trainer_state_dict["cpu_rng_state"])
        except:
            logger.warning("Cannot restore CPU random state!")

        try:
            # sometimes we have less number of devices
            trainer_state_dict["cuda_rng_state"] = trainer_state_dict["cuda_rng_state"][:torch.cuda.device_count()]
            torch.cuda.set_rng_state_all(trainer_state_dict["cuda_rng_state"])
        except (IndexError, RuntimeError):
            # if we still cannot load back cuda rng_states, we ignore it
            logger.warning("Cannot restore CUDA random state!")

        try:
            # restore amp states
            if self.config.training.fp16:
                amp.load_state_dict(trainer_state_dict["amp_state_dict"])
        except:
            logger.warning("Cannot restore AMP state!")

    def load_amp_state_dict(self, trainer_state_dict):
        try:
            # restore amp state dict
            if self.config.training.fp16:
                amp.load_state_dict(trainer_state_dict["amp_state_dict"])
        except:
            logger.warning("Cannot restore AMP state!")


    def __del__(self):
        if self.config.training.num_gpus_per_node > 1:
            for p in self._spawn_context.processes:
                p.kill()

# def set_random_port(self):
#     """
#     When running DDP NOT managed by SLURM, the ports might collide
#     :return:
#     """
#     try:
#         default_port = os.environ['MASTER_PORT']
#     except Exception:
#         import random
#         default_port = random.randint(20000, 29000)
#         os.environ['MASTER_PORT'] = str(default_port)
