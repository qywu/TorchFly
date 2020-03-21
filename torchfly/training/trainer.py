import os
import sys
import ray
import math
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
from torchfly.training.callbacks import TrainHandler, LogHandler, GradientClipNorm, Checkpoint
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
        self.total_num_steps = int(self.config.training.total_num_steps)
        self.total_num_epochs = int(self.config.training.total_num_epochs)

        self.callback_handler = CallbackHandler(
            config, trainer=self, callbacks=self.default_callbacks(), verbose=config.logging.level == "DEBUG"
        )

    def default_callbacks(self) -> List:
        callbacks = []
        callbacks.append(TrainHandler(self.config))
        callbacks.append(LogHandler(self.config))
        callbacks.append(GradientClipNorm(self.config))
        callbacks.append(Checkpoint(self.config))
        return callbacks

    def train(self) -> Dict[str, Any]:

        self.callback_handler.fire_event(Events.INITIALIZE)

        if self.config.training.num_gpus_per_node > 1:
            logger.info("Initializing Distributed Training")

            if 'OMP_NUM_THREADS' not in os.environ:
                os.environ["OMP_NUM_THREADS"] = str(4)
                logger.info(
                    "\n*****************************************\n"
                    "Setting OMP_NUM_THREADS environment variable for each process \n"
                    "to be {} in default, to avoid your system being overloaded, \n"
                    "please further tune the variable for optimal performance in \n"
                    "your application as needed. \n"
                    "*****************************************\n".format(os.environ["OMP_NUM_THREADS"])
                )

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(random.randint(20000, 29000))  # use a random port, but might collide
            os.environ["WORLD_SIZE"] = str(self.config.training.num_gpus_per_node)

            torch.multiprocessing.set_start_method('spawn')
            # multiprocessing.log_to_stderr()
            # TODO: Use Process instead of spawn
            torch.multiprocessing.spawn(self._train, args=(), nprocs=self.config.training.num_gpus_per_node)
            results = {}
        elif self.config.training.num_gpus_per_node == 1:
            logger.info("Initializing Single GPU Training")
            results = self._train(rank=0)
        else:
            raise NotImplementedError("Do you mean CPU training?")

        return results

    def _train(self, rank=0):
        self.rank = rank
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
            if (self.global_step_count + 1) % self.config.training.gradient_accumulation_steps == 0:
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

        if self.config.training.gradient_accumulation_steps > 1:
            loss = loss / self.config.training.gradient_accumulation_steps

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

    def configure_optimizer(self) -> Optimizer:
        no_decay = ["bias", "LayerNorm.weight", "ln", "Norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]

        if self.config.training.optimizer == "AdamW":
            return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.training.learning_rate)
        elif self.config.training.optimizer == "FusedAdam":
            return apex.optimizers.FusedAdam(optimizer_grouped_parameters, lr=self.config.training.learning_rate)
        elif self.config.training.optimizer == "Adadelta":
            return torch.optim.Adadelta(optimizer_grouped_parameters, lr=self.config.training.learning_rate)
        elif self.config.training.optimizer == "FusedLAMB":
            return apex.optimizers.FusedLAMB(optimizer_grouped_parameters, lr=self.config.training.learning_rate)
        else:
            raise NotImplementedError

    def configure_scheduler(self) -> LambdaLR:
        if self.config.training.scheduler == "Constant":
            return ConstantLRSchedule(self.optimizer)
        elif self.config.training.scheduler == "WarmupConstant":
            return WarmupConstantSchedule(self.optimizer, self.config.training.warmup_steps)
        elif self.config.training.scheduler == "WarmupLinear":
            return WarmupLinearSchedule(self.optimizer, self.config.training.warmup_steps, self.total_num_steps)
        elif self.config.training.scheduler == "WarmupCosine":
            if self.config.traing.warmup_cosine_cycle is None:
                cycles = 0.5
            else:
                cycles = self.config.traing.warmup_cosine_cycle
            return WarmupCosineSchedule(
                self.optimizer, self.config.training.warmup_steps, self.total_num_steps, cycles=cycles
            )
        elif self.config.training.scheduler == "WarmupCosineWithHardRestartsSchedule":
            if self.config.traing.warmup_cosine_cycle is None:
                cycles = 0.5
            else:
                cycles = self.config.traing.warmup_cosine_cycle
            return WarmupCosineWithHardRestartsSchedule(
                self.optimizer, self.config.training.warmup_steps, self.total_num_steps, cycles=cycles
            )
        else:
            logger.error("Write your own version of `configure_scheduler`!")
            raise NotImplementedError

    def state_dict(self):
        # TODO: add error handling for saving and loading rng states and amp states
        # which can simply be loading the first device's rng states
        if self.config.training.num_gpus_per_node > 1:
            model_states = self.model.module.state_dict()
        else:
            model_states = self.model.state_dict()

        states = {
            "epoch": self.epochs_trained,
            "iteration": self.global_step_count,
            "iteration_in_epoch": self.local_step_count,
            "model_states": model_states,
            "optimizer_states": self.optimizer.state_dict(),
            "scheduler_states": self.scheduler.state_dict(),
            "cpu_rng_states": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        # save amp states
        if self.config.training.fp16:
            states["amp"] = amp.state_dict()
        return states

    def load_state_dict(self, states: Dict[str, Any]):
        self.epochs_trained = states["epoch"]
        self.global_step_count = states["iteration"]
        self.local_step_epoch = states["iteration_in_epoch"]
        if self.config.training.num_gpus_per_node > 1:
            self.model.module.load_state_dict(states["model_states"])
        else:
            self.model.load_state_dict(states["model_states"])
        self.optimizer.load_state_dict(states["optimizer_states"])
        self.scheduler.load_state_dict(states["scheduler_states"])
        # cpu random state
        torch.set_rng_state(states["cpu_rng_states"])
        # sometimes we have less number of devices
        states["cuda_rng_states"] = states["cuda_rng_states"][:torch.cuda.device_count()]
        try:
            torch.cuda.set_rng_state_all(states["cuda_rng_states"])
        except (IndexError, RuntimeError):
            # if we still cannot load back cuda rng_states, we ignore it
            logger.warn("Cannot restore CUDA random state!")
            pass
        # restore amp states
        if self.config.training.fp16:
            amp.load_state_dict(states["amp"])


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
