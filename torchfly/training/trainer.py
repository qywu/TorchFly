from typing import Any, List, Dict, Iterator, Callable, Iterable
import os
import sys
import torch
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf
import socket
# import apex
# from apex import amp
# from apex.parallel import DistributedDataParallel, Reducer
# from torch.nn.parallel import DistributedDataParallel

# local imports
from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.training.callbacks import Checkpoint, Evaluation, Console, Resume
from torchfly.flylogger.train_logger import TrainLogger
from torchfly.utilities import move_to_device
from torchfly.training import FlyModel
from torchfly.training.reducer import Reducer

import logging

logger = logging.getLogger(__name__)

# pylint: disable=no-member


class Trainer:

    def __init__(self, config: DictConfig, model: FlyModel, name: str = "task1", *args, **kwargs):
        """
        Args:
            config: FlyConfig dictionary
            model: must be FlyModel
            dataloader_fn: a Callable function which returns dataloaders
        """
        logger.info("TrainerLoop is initializing!")
        if not isinstance(model, FlyModel):
            logger.warn("model is not defined as FlyModel")
        self.config = config
        self.model = model
        self.name = name

        # class properties
        self.rank = None
        self.local_rank = None
        self.node_rank = None
        self.world_size = None
        self.distributed_training = None
        self.device = None
        self.fp16 = config.fp16
        self.gradient_accumulation_batches = config.gradient_accumulation_batches
        self.callback_handler = None
        self.optimizers = []
        self.schedulers = []

        self.init_distributed_environment()

        # Model is sent to GPU or CPU
        self.init_device()
        # self.optimizers, self.schedulers = self.configure_optimizers()

        self.model = move_to_device(self.model, self.device)
        self.model.device = self.device
        self.init_fp16()

        if self.distributed_training:
            self.init_distributed_model(self.model)

        # make sure the model has access to trainer info
        self.model.set_trainer(self)

        self.callback_handler = CallbackHandler(config,
                                                trainer=self,
                                                callbacks=[],
                                                verbose=config.logging.level == "DEBUG")

        # Configure all callbacks
        self.configure_callbacks()
        self.callback_handler.fire_event(Events.INITIALIZE)

    def init_distributed_environment(self):
        # For distributed
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.distributed_training = (self.world_size > 1)

        # TODO: add error message when num_gpus is set, but distributed training is False here

        if self.distributed_training and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            assert torch.distributed.is_initialized()

        if self.distributed_training and not torch.distributed.is_initialized():
            self.node_rank = os.environ.get("NODE_RANK", "N/A")
            logger.info(
                f"Initialized Rank:{torch.distributed.get_rank()} Locak-rank: {self.local_rank} on Node:{self.node_rank} Node-name:{socket.gethostname()}"
            )

    def init_device(self):
        # set cuda device
        if self.config.num_gpus_per_node > 0:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

    def init_fp16(self):
        if self.config.num_gpus_per_node == 0:
            raise NotImplementedError("For mixed precision training, you need to use GPU!")
        self.loss_scaler = GradScaler()

    def init_training_constants(self):
        self.total_num_update_steps = int(self.config.total_num.update_steps)
        self.total_num_batches = self.total_num_update_steps * int(self.gradient_accumulation_batches)
        self.total_num_epochs = int(self.config.total_num.epochs)

        # check if training in epoch or update_steps
        if self.total_num_update_steps < 0 and self.total_num_epochs < 0:
            raise NotImplementedError("config.total_num.updated_steps must be larger than 0")
        elif self.total_num_update_steps > 0 and self.total_num_epochs > 0:
            raise NotImplementedError(
                "Please only set either config.total_num.updated_steps or config.total_num.epochs greater than 0")
        elif self.total_num_update_steps > 0 and self.total_num_epochs < 0:
            self.training_in_epoch = False
        elif self.total_num_update_steps < 0 and self.total_num_epochs > 0:
            self.training_in_epoch = True

        # get the number of batches in the dataloader for one epoch
        try:
            self.epoch_num_batches = len(self.train_dataloader)
        except TypeError:
            logger.warning("Cannot determine the length of train_dataloader!")
            self.epoch_num_batches = None

        if self.training_in_epoch:
            if self.epoch_num_batches is not None:
                self.total_num_batches = self.epoch_num_batches * self.total_num_epochs
                self.total_num_update_steps = self.total_num_batches // self.gradient_accumulation_batches
                self.epoch_num_update_steps = self.epoch_num_batches // self.gradient_accumulation_batches
            else:
                # this is set to wait until the epoch finishes first
                self.total_num_update_steps = sys.maxsize

    def configure_optimizers(self, total_num_update_steps=None, optimizers=None, schedulers=None):
        if optimizers is not None and schedulers is not None:
            self.optimizers, self.schedulers = optimizers, schedulers
        elif total_num_update_steps is not None:
            self.optimizers, self.schedulers = self.model.configure_optimizers(total_num_update_steps)
        else:
            raise ValueError("Please provide the correct argument!")
        return self.optimizers, self.schedulers

    def configure_callbacks(self):
        # Resume callback runs for all ranks
        if self.config.resume.enabled:
            self.resume_callback = Resume(self.config)
            self.add_callback(self.resume_callback)

        self.log_callback = TrainLogger(self.config)
        self.add_callback(self.log_callback)

        self.eval_callback = Evaluation(self.config)
        self.add_callback(self.eval_callback)

        # For logging and inference, use rank 0 by default
        if self.rank == 0:
            if self.config.console:
                self.console_callback = Console(self.config)
                self.add_callback(self.console_callback)

            if self.config.checkpointing.enabled:
                self.checkpoint_callback = Checkpoint(self.config)
                self.add_callback(self.checkpoint_callback)

    def init_distributed_model(self, model):
        """
        Default distributed training uses reducer for simplicity. 
        """
        # Distributed training (should be after apex fp16 initialization)
        self.reducer = Reducer(model)
        # for param in self.model.parameters():
        #     dist.broadcast(param.data, 0)

    def train(self,
              train_dataloader,
              validation_dataloader=None,
              test_dataloader=None,
              configure_optimizers=True,
              name=None,
              *args,
              **kwargs):
        self.total_num_update_steps = 0
        self.total_num_batches = 0
        self.total_num_epochs = 0
        self.epoch_num_batches = 0
        self.global_batch_count = 0
        self.global_step_count = 0
        self.epochs_trained = 0
        self.local_step_count = 0

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.init_training_constants()

        if configure_optimizers or len(self.optimizers) == 0:
            self.configure_optimizers(self.total_num_update_steps)

        if name is not None:
            self.name = name

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
                if self.global_step_count < self.total_num_update_steps:
                    continue
                else:
                    break

        # Training ends
        self.callback_handler.fire_event(Events.TRAIN_END)

    def train_epoch(self):
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0]

        self.local_step_count = 0

        if self.train_dataloader is None:
            return

        for batch in self.train_dataloader:
            self.callback_handler.fire_event(Events.BATCH_BEGIN)

            batch = move_to_device(batch, self.device)
            output = self.backward_batch(batch)

            # Update the model
            if (self.global_batch_count + 1) % self.gradient_accumulation_batches == 0:
                # Update the model with optimizer
                self.step_update(self.model, self.optimizer, self.scheduler)
                self.global_step_count += 1
                self.local_step_count += 1

            self.callback_handler.fire_event(Events.BATCH_END)

            if self.global_step_count >= self.total_num_update_steps:
                break

            self.global_batch_count += 1

    def backward_batch(self, batch):
        self.model.train()
        with torch.cuda.amp.autocast(self.fp16):
            output = self.model(batch)

        # get the loss from output
        if hasattr(output, "loss"):
            loss = output.loss
        elif isinstance(output, dict):
            loss = output["loss"]

        if self.gradient_accumulation_batches > 1:
            loss = loss / self.gradient_accumulation_batches

        self.loss_backward(loss)
        return output

    def step_update(self, model, optimizer, scheduler=None):
        """
            self.loss_scaler is defined in `configure_fp16`
        """
        self.callback_handler.fire_event(Events.STEP_BEGIN)
        # collect gradient
        if self.distributed_training:
            self.reducer.reduce()

        gradient_clip = self.config.optimization.max_gradient_norm
        # Gradient Clipping
        if gradient_clip > 0:
            if self.fp16:
                self.loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        # Update the model
        if self.fp16:
            self.loss_scaler.step(optimizer)
            self.loss_scaler.update()
        else:
            optimizer.step()
        # Step learning rate
        if scheduler:
            scheduler.step()
        # Gradient to zero
        optimizer.zero_grad()
        self.callback_handler.fire_event(Events.STEP_END)

    def loss_backward(self, loss):
        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
        # Loss backward
        if self.fp16:
            self.loss_scaler.scale(loss).backward()
        else:
            loss.backward()
        self.callback_handler.fire_event(Events.BACKWARD_END)

    def validate(self, dataloader):
        # Start Validation
        self.model.reset_evaluation_metrics()
        self.callback_handler.fire_event(Events.VALIDATE_BEGIN)
        self.model.validation_loop(dataloader)
        self.callback_handler.fire_event(Events.VALIDATE_END)

    def test(self, dataloader):
        # Start Testing
        self.model.reset_evaluation_metrics()
        self.callback_handler.fire_event(Events.TEST_BEGIN)
        self.model.test_loop(dataloader)
        self.callback_handler.fire_event(Events.TEST_END)

    def set_model_state(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)

    def get_model_state(self):
        return self.model.state_dict()

    def set_trainer_state(self, trainer_state_dict):
        self.epochs_trained = trainer_state_dict["epochs_trained"]
        self.global_step_count = trainer_state_dict["global_step_count"]
        self.local_step_count = trainer_state_dict["local_step_count"]

        # Resume the training state
        if self.config.resume.resume:
            # Scheduler States
            if self.config.resume.resume_scheduler:
                for idx, scheduler in enumerate(self.schedulers):
                    try:
                        scheduler.load_state_dict(trainer_state_dict["schedulers_state_dict"][idx])
                    except:
                        if self.rank == 0:
                            logger.warning(f"Cannot Load Scheduler {idx}'s State!")

            if self.config.resume.resume_optimizer:
                for idx, optimizer in enumerate(self.optimizers):
                    try:
                        optimizer.load_state_dict(trainer_state_dict["optimizers_state_dict"][idx])
                    except:
                        if self.rank == 0:
                            logger.warning(f"Cannot Load Optimizer {idx}'s State!")

            # save amp states
            if self.fp16:
                self.loss_scaler.load_state_dict(trainer_state_dict["amp_state_dict"])

            # Random States
            if self.config.resume.resume_rng_state:
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
        if self.fp16:
            trainer_state_dict["amp_state_dict"] = self.loss_scaler.state_dict()

        # All Callbacks
        for callback in self.callback_handler.callbacks:
            trainer_state_dict[str(type(callback))] = callback.state_dict()

        return trainer_state_dict

    def add_callback(self, callback: Callback):
        self.callback_handler.add_callback(callback)


# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']

# def get_log_variable(x):
#     if isinstance(x, torch.Tensor):
#         x = x.detach()
#         return x.item()
#     else:
#         raise NotImplementedError