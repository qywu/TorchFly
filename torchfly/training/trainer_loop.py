from typing import Any, List, Dict, Iterator, Callable, Iterable
import os
import random
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig
import socket
import apex
from apex import amp
from apex.parallel import DistributedDataParallel, Reducer
# from torch.nn.parallel import DistributedDataParallel

# local imports
from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.training.callbacks import LogHandler, Checkpoint, Evaluation
from torchfly.common import move_to_device
from torchfly.training import FlyModel

import logging

logger = logging.getLogger(__name__)

# pylint: disable=no-member


class TrainerLoop:
    def __init__(
        self,
        config: DictConfig,
        model: FlyModel,
        train_dataloader_fn: Callable,
        valid_dataloader_fn: Callable = None,
        test_dataloader_fn: Callable = None
    ):
        """
        Args:
            config: FlyConfig dictionary
            model: must be FlyModel
            dataloader_fn: a Callable function which returns dataloaders
        """
        assert isinstance(model, FlyModel)

        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Distributed
        if self.world_size > 1:
            # Initialize distributed training
            torch.distributed.init_process_group(
                backend="nccl", rank=self.rank, world_size=self.world_size, init_method='env://'
            )
            self.node_rank = os.environ.get("NODE_RANK", "UNK")
            print(
                f"Initialized Rank:{dist.get_rank()} Locak-rank: {self.local_rank} on Node:{self.node_rank} Node-name:{socket.gethostname()}"
            )

        print("Starting")

        # configure distributed training
        self.model = model

        self.train_dataloader = train_dataloader_fn(config)

        if self.rank == 0:
            self.validation_dataloader: Iterable = valid_dataloader_fn(config) if valid_dataloader_fn else None
            self.test_dataloader = test_dataloader_fn(config) if test_dataloader_fn else None

        self.callback_handler = CallbackHandler(
            config, trainer=self, callbacks=[], verbose=config.training.logging.level == "DEBUG"
        )

        # constants
        self.gradient_accumulation_steps = config.training.optimization.gradient_accumulation_steps
        self.fp16 = config.training.optimization.fp16
        # self.fp16_opt_level = config.training.optimization.fp16_opt_level
        self.distributed_training = False

        self.total_num_update_steps = int(config.training.total_num.update_steps)
        self.total_num_steps = self.total_num_update_steps * int(self.gradient_accumulation_steps)

        self.total_num_epochs = int(self.config.training.total_num.epochs)

        # Train in epochs or steps
        if self.total_num_epochs > 0:
            self.training_in_epoch = True
        else:
            if self.total_num_update_steps < 0:
                raise NotImplementedError("config.training.total_num.updated_steps must be larger than 0")
            self.training_in_epoch = False
            self.total_num_epochs = 1

        # Number of training batches
        if self.training_in_epoch:
            try:
                self.epoch_num_training_steps = len(self.train_dataloader)
                self.total_num_training_steps = self.epoch_num_training_steps * self.total_num_epochs
                self.total_num_update_steps = self.total_num_training_steps // self.gradient_accumulation_steps
                self.total_num_steps = self.total_num_training_steps
            except TypeError:
                # connot set the number of total_num_epoch
                # because it is impossible to know
                logger.error("Cannot get the length of train dtrainer.model")
                raise NotImplementedError("Please specify the `total_num_epochs` or `total_num_update_steps`!")
        else:
            self.epoch_num_training_steps = self.total_num_update_steps

        # local variables
        self.global_step_count = 0
        self.epochs_trained = 0
        self.local_step_count = 0

        # set cuda device
        if config.training.num_gpus_per_node > 0:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        # Configure optimizers
        self.optimizers, self.schedulers = self.model.configure_optimizers(self.total_num_update_steps)
        self.optimizers, self.schedulers = self.configure_optimizers()

        # Model is sent to GPU or CPU
        self.model = move_to_device(self.model, self.device)

        # Mixed-Precision
        if self.fp16:
            if self.config.training.num_gpus_per_node == 0:
                raise NotImplementedError("For mixed precision training, you need to use GPU!")
            self.configure_fp16()

        # Distributed Training
        if self.world_size > 1:
            self.configure_ddp()

        self.configure_callbacks()

        self.log_keys = set()
        self.tmp_vars = {}
        self.callback_handler.fire_event(Events.INITIALIZE)

        # make sure the model has access to trainer info
        self.model.set_trainer(self)

    def update_log_keys(self, keys: List[str]):
        self.log_keys.update(keys)

    def configure_optimizers(self):
        return self.model.configure_optimizers(self.total_num_update_steps)

    def configure_callbacks(self):
        # Callback
        # by default set up LogHandler and Checkpointer
        self.checkpoint_callback = Checkpoint(self.config)
        self.add_callback(self.checkpoint_callback)

        # For logging and inference, use rank 0 by default
        if self.rank == 0:
            self.log_callback = LogHandler(self.config)
            self.add_callback(self.log_callback)

            self.inference_callback = Inference(self.config)
            self.add_callback(self.inference_callback)

    def configure_fp16(self):
        self.loss_scaler = GradScaler()

    def configure_ddp(self):
        # Distributed training (should be after apex fp16 initialization)
        self.distributed_training = True
        self.model = DistributedDataParallel(self.model, delay_allreduce=True)
        # trainer.model = torch.nn.parallel.DistributedDataParallel(
        #     trainer.model, device_ids=[trainer.rank], output_device=trainer.rank, find_unused_parameters=True
        # )

    def train(self):
        # Training begins
        self.callback_handler.fire_event(Events.TRAIN_BEGIN)

        while True:
            self.callback_handler.fire_event(Events.EPOCH_BEGIN)
            self.train_epoch()
            self.callback_handler.fire_event(Events.EPOCH_END)
            self.epochs_trained += 1

            if self.training_in_epoch:
                if self.epochs_trained >= self.total_num_epochs:
                    break
            else:
                if self.global_step_count < self.total_num_steps:
                    continue
                else:
                    break

        # Training ends
        self.callback_handler.fire_event(Events.TRAIN_END)

    def train_epoch(self):
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0]

        self.local_step_count = 0

        for batch in self.train_dataloader:
            self.callback_handler.fire_event(Events.BATCH_BEGIN)

            batch = move_to_device(batch, self.device)
            self.tmp_vars["log_dict"] = self.train_step(batch)

            # Update the model
            if (self.global_step_count + 1) % self.gradient_accumulation_steps == 0:
                self.step_update()

            self.callback_handler.fire_event(Events.BATCH_END)

            if self.global_step_count >= self.total_num_steps:
                break

            if self.config.training.num_gpus_per_node > 1:
                torch.distributed.barrier()

            self.global_step_count += 1
            self.local_step_count += 1

    def step_update(self):
        self.callback_handler.fire_event(Events.STEP_BEGIN)

        if self.fp16:
            self.loss_scaler.unscale_(self.optimizer)

        if self.config.training.optimization.max_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.optimization.max_gradient_norm)

        if self.fp16:
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        self.callback_handler.fire_event(Events.STEP_END)

    def train_step(self, batch):
        self.optimizer = self.optimizers[0]

        if self.fp16:
            with autocast():
                results = self.model(batch)
        else:
            results = self.model(batch)

        loss = results["loss"]

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
        self.loss_backward(loss)
        self.callback_handler.fire_event(Events.BACKWARD_END)
        # return the results

        log_dict = {"loss": loss.item() * self.gradient_accumulation_steps}
        log_dict["_lr"] = get_lr(self.optimizer)

        for key in self.log_keys:
            log_dict[key] = get_log_variable(results[key])

        return log_dict

    def loss_backward(self, loss):
        # Loss backward
        if self.fp16:
            self.loss_scaler.scale(loss).backward()
        else:
            loss.backward()

    def validate(self):
        # Start Validation
        self.callback_handler.fire_event(Events.VALIDATE_BEGIN)
        # Validation
        self.model.eval()
        # No gradient is needed for validation
        with torch.no_grad():
            pbar = tqdm.tqdm(self.validation_dataloader)
            pbar.mininterval = 2.0
            for batch in pbar:
                # send to cuda device
                batch = move_to_device(batch, self.device)

                if self.distributed_training:
                    self.model.module.predict(batch)
                else:
                    self.model.predict(batch)

        self.callback_handler.fire_event(Events.VALIDATE_END)

    def test(self):
        # Start Testing
        self.callback_handler.fire_event(Events.TEST_BEGIN)
        # Validation
        self.model.eval()
        # No gradient is needed for test
        with torch.no_grad():
            pbar = tqdm.tqdm(self.test_dataloader)
            pbar.mininterval = 2.0
            for batch in pbar:
                # send to cuda device
                batch = move_to_device(batch, self.device)

                if self.distributed_training:
                    self.model.module.predict(batch)
                else:
                    self.model.predict(batch)
        self.callback_handler.fire_event(Events.TEST_END)

    def set_model_state(self, model_state_dict):
        if self.distributed_training:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

    def get_model_state(self):
        if self.distributed_training:
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()

    def set_trainer_state(self, trainer_state_dict):
        self.epochs_trained = trainer_state_dict["epochs_trained"]
        self.global_step_count = trainer_state_dict["global_step_count"]
        self.local_step_count = trainer_state_dict["local_step_count"]

        # Resume the training state
        if self.config.training.resume.resume:
            # Scheduler States
            if self.config.training.resume.resume_schedulers:
                for idx, scheduler in enumerate(self.schedulers):
                    try:
                        scheduler.load_state_dict(trainer_state_dict["schedulers_state_dict"][idx])
                    except:
                        if self.rank == 0:
                            logger.warning(f"Cannot Load Scheduler {idx}'s State!")

            # save amp states
            if self.config.training.optimization.fp16:
                self.loss_scaler.load_state_dict(trainer_state_dict["amp_state_dict"])

            # Random States
            if self.config.training.resume.resume_rng_state:
                torch.set_rng_state(trainer_state_dict["cpu_rng_state"])
                trainer_state_dict["cuda_rng_state"] = trainer_state_dict["cuda_rng_state"][:torch.cuda.device_count()]
                torch.cuda.set_rng_state_all(trainer_state_dict["cuda_rng_state"])

            # All Callbacks
            for callback in self.callback_handler.callbacks:
                try:
                    callback.load_state_dict(trainer_state_dict[str(type(callback))])
                except:
                    logger.error(f"{type(callback)} seems not to exist in the checkpoint state!")

    def get_trainer_state(self):
        trainer_state_dict = {
            "epochs_trained": self.epochs_trained,
            "global_step_count": self.global_step_count,
            "local_step_count": self.local_step_count,
            "optimizers_state_dict": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers_state_dict": [scheduler.state_dict() for scheduler in self.schedulers],
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }

        # save amp states
        if self.config.training.optimization.fp16:
            trainer_state_dict["amp_state_dict"] = self.loss_scaler.state_dict()

        # All Callbacks
        for callback in self.callback_handler.callbacks:
            trainer_state_dict[str(type(callback))] = callback.state_dict()

        return trainer_state_dict

    def add_callback(self, callback: Callback):
        self.callback_handler.add_callback(callback)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_log_variable(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        return x.item()
    else:
        raise NotImplementedError