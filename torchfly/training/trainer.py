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
from apex import amp
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import torch.multiprocessing as multiprocessing
import torch.distributed
from apex.parallel import DistributedDataParallel, Reducer
# from torch.nn.parallel import DistributedDataParallel

from .checkpointer import Checkpointer
from .optimization import ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule
from ..common import move_to_device, configure_logging

logger = logging.getLogger(__name__)


class MovingAverage:
    def __init__(self, init_dict=None, decay=0.9):
        self.history_dict = init_dict if init_dict else {}
        self.decay = decay

    def update_key(self, key, value):
        if key in self.history_dict:
            self.history_dict[key] = self.decay * self.history_dict[key] + (1 - self.decay) * value
        else:
            self.history_dict[key] = value
        return self.history_dict[key]


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module = None,
        train_loader: Iterator = None,
        validation_loader: Iterator = None,
        test_loader: Iterator = None,
        train_loader_fn: Callable = None,
    ):
        """
        Do not send anything to cuda in __init__
        """
        self.config = config
        self.moving_average = MovingAverage()

        # # Data Loading
        self.train_loader = train_loader
        self.train_loader_fn = train_loader_fn
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        # local variables
        self.__global_count = 0
        self.__epochs_trained = 0
        self.__count_in_epoch = 0

        # setup model
        if model is None:
            self.model = self.configure_model()
        else:
            self.model = model

        # setup the scheduler
        self.optimizer = self.configure_optimizer()

        if self.config.saving.save_dir is None:
            # Saving directory
            self.config.saving.save_dir = os.path.join(os.getcwd(), "Checkpoints")
        else:
            self.config.saving.save_dir = os.path.join(os.getcwd(), self.config.saving.save_dir)

        # # Checkpointer
        self.checkpointer = Checkpointer(
            sync_every_save=True,
            num_checkpoints_to_keep=config.saving.num_checkpoints_to_keep,
            keep_checkpoint_every_num_seconds=config.saving.keep_checkpoint_every_num_seconds,
            storage_dir=self.config.saving.save_dir
        )

        # Logging
        self.__log_in_seconds = False
        if self.config.logging.iterations_interval <= 0:
            if self.config.logging.seconds_interval is None or self.config.logging.seconds_interval < 0:
                # default log_iterations_interval
                self.config.logging.iterations_interval = 10
            else:
                self.__log_in_seconds = True

        # Saving
        self.__save_in_seconds = False
        # if nothign about saving interval is set
        if (self.config.saving.iterations_interval is None and self.config.saving.seconds_interval is None
           ) or (self.config.saving.iterations_interval < 0 and self.config.saving.seconds_interval < 0):
            # then save for every epoch
            self.config.saving.iterations_interval = len(self.train_loader) - 1

        if self.config.saving.iterations_interval < 0 and self.config.saving.seconds_interval > 0:
            self.__save_in_seconds = True

        logger.info(self.config.pretty())

        # # Resume the training
        # # find the any checkpoint
        if self.config.training.resume:
            self.__cache_states = self.__restore_checkpoint()

    def __setup_configuration(self):
        # Reproducibility
        if self.config.training.random_seed:
            self.__set_random_seed(self.__rank + self.config.training.random_seed)

        # DataLoader
        if self.train_loader is None:
            if self.train_loader_fn is None:
                logger.error("Please specify either `train_loader` or `train_loader_fn`!")
                raise NotImplementedError
            self.train_loader = self.train_loader_fn(self.config)

        # Set Num of Epochs and Num of Iterations
        # if the num of epochs is set
        if self.config.training.total_num_epochs > 0:
            self.__total_num_iterations = len(self.train_loader) * self.config.training.total_num_epochs
            self.__total_num_epochs = self.config.training.total_num_epochs
        # num of epochs is not set
        else:
            if self.config.training.total_num_iterations is None:
                raise NotImplementedError("Please specify the `total_num_epochs` or `total_num_iterations`!")
            else:
                self.__total_num_epochs = self.__total_num_iterations // len(self.train_loader)

        # Setup validation interval
        if self.config.training.validation_iterations_interval is None or \
            self.config.training.validation_iterations_interval < 0:
            # validation for every epoch
            self.config.training.validation_iterations_interval = len(self.train_loader) - 1

    def __single_gpu_train(self):
        self.__rank = 0
        self.__master = True
        self.__device = torch.device("cuda")

        self.__setup_configuration()
        self.__setup_training()

        epoch_results = self.__train()

        return epoch_results

    def _distributed_train(self, rank=0):
        """
        Handles the distributed training on a single node.

        Args:
            rank: also the gpu index
        """

        self.__rank = rank
        self.__master = rank == 0
        torch.cuda.set_device(rank)
        self.__device = torch.device("cuda", self.__rank)
        self.config.training.rank = rank

        # init distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.__rank, world_size=self.config.training.num_gpus_per_node
        )

        self.__setup_configuration()
        self.__setup_training()

        # Distributed training (should be after apex fp16 initialization)
        self.model = DistributedDataParallel(self.model)

        self.__train()
        return None

    def __setup_training(self) -> None:
        if self.__master:
            # close existing logging
            for _handler in logger.handlers[:]:
                _handler.close()
                logger.removeHandler(_handler)
            for _filter in logger.filters[:]:
                logger.removeFilter(_filter)

            configure_logging(self.config)
            
            if not ray.is_initialized():
                logger.info(ray.init())
            self.__tensorboard = SummaryWriter(log_dir=os.getcwd(), purge_step=self.__global_count)

        # To Device
        self.model = move_to_device(self.model, self.__device)

        # FP16
        if self.config.training.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.config.training.fp16_opt_level
            )

        # Scheduler
        self.scheduler = self.configure_scheduler()

        # Resume the training
        if self.__cache_states is not None:
            self.load_state_dict(self.__cache_states)
            logger.info(f"Loaded the saved checkpoint {self.__cache_states['file_path']}")
        else:
            logger.info("Not loading any checkpoint. Training from scratch!")

    def __train(self):
        if self.__master:
            logger.info("Starting Training.")
            if self.__log_in_seconds:
                self.__log_iter_start_time = time.time()
            if self.__save_in_seconds:
                self.__save_iter_start_time = time.time()
            epoch_start_time = time.time()

        if self.__master:
            if self.__global_count > 0:
                resume_training = True
                logger.info("Resume the training!")
            else:
                resume_training = False

        for _ in range(self.__global_count, self.__total_num_iterations):
            if self.__master:
                if self.__count_in_epoch == 0:
                    logger.info("Epoch %d/%d", self.__epochs_trained + 1, self.__total_num_epochs)
                    epoch_start_time = time.time()

            # The state should be perserved by torch.get_rng_state
            # However, this solution is not deterministic, but at least it ensures
            # the randomness when loading data
            batch = next(self.train_loader)

            batch = move_to_device(batch, self.__device)
            results = self.train_iter(batch)

            # Update the model
            if (self.__global_count + 1) % self.config.training.gradient_accumulation_steps == 0:
                # gradient norm clipping
                if self.config.training.max_gradient_norm > 0:
                    if self.config.training.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), self.config.training.max_gradient_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_gradient_norm)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Checkpointing
            if self.__master:
                if self.__save_in_seconds:
                    current_time = time.time()
                    iter_elapsed_time = current_time - self.__save_iter_start_time

                    if iter_elapsed_time > self.config.saving.seconds_interval:
                        self.__save_checkpoint()
                        self.__save_iter_start_time = current_time
                else:
                    if (self.__global_count + 1) % self.config.saving.iterations_interval == 0:
                        self.__save_checkpoint()

            # Logging
            if self.__master:
                if resume_training:
                    self.__log_iteration(self.__count_in_epoch, results)
                    resume_training = False
                elif self.__log_in_seconds:
                    current_time = time.time()
                    iter_elapsed_time = current_time - self.__log_iter_start_time

                    if iter_elapsed_time > self.config.logging.seconds_interval:
                        self.__log_iteration(self.__count_in_epoch, results)
                        self.__log_iter_start_time = current_time
                        
                else:
                    if self.__global_count % self.config.logging.iterations_interval == (
                        self.config.logging.iterations_interval - 1
                    ):
                        self.__log_iteration(self.__count_in_epoch, results)

            # Post
            self.__count_in_epoch += 1
            self.__global_count += 1

            # reset the counters
            if self.__count_in_epoch >= len(self.train_loader):
                self.__epochs_trained += 1
                self.__count_in_epoch = 0

                if self.__master:
                    epoch_elapsed_time = time.time() - epoch_start_time
                    logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            # Validation
            if self.__master:
                if self.__global_count % self.config.training.validation_iterations_interval == 0:
                    if self.validation_loader:
                        self.model.eval()
                        logger.info("Validation Starts!")
                        self.validate()
                        self.model.train()

        if self.__master:
            self.__tensorboard.close()

        return {}

    def train(self) -> Dict[str, Any]:
        if self.config.training.num_gpus_per_node > 1:
            logger.info("Initializing Distributed Training")

            if 'OMP_NUM_THREADS' not in os.environ and self.config.training.num_gpus_per_node > 1:
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
                multiprocessing.spawn(self._distributed_train, args=(), nprocs=self.config.training.num_gpus_per_node)
        elif self.config.training.num_gpus_per_node == 1:
            self.__single_gpu_train()
        else:
            raise NotImplementedError("Do you mean CPU training?")

        results = {}
        return results

    def train_iter(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        results = self.model(batch)
        loss = results["loss"]

        if self.config.training.gradient_accumulation_steps > 1:
            loss = loss / self.config.training.gradient_accumulation_steps

        if self.config.training.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return results

    def validate(self):
        self.model.eval()
        for batch in self.validation_loader:
            # send to cuda device
            batch = move_to_device(batch, self.__device)
            with torch.no_grad():
                if self.config.training.num_gpus_per_node > 1:
                    self.model.module.predict(batch)
                else:
                    self.model.predict(batch)

        if self.config.training.num_gpus_per_node > 1:
            metrics = self.model.module.get_metrics(reset=True)
        else:
            metrics = self.model.get_metrics(reset=True)

        for metric_name, value in metrics.items():
            metric_name = metric_name[0].upper() + metric_name[1:]
            logger.info(f"Epoch {self.__epochs_trained}: Validation {metric_name} {value:4.4f}")
            if isinstance(value, float):
                self.__tensorboard.add_scalar(metric_name, value, global_step=self.__global_count)

    def configure_optimizer(self) -> Optimizer:
        if self.config.training.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.training.learning_rate)
        elif self.config.training.optimizer == "Adadelta":
            return torch.optim.Adadelta(self.model.parameters(), lr=self.config.training.learning_rate)
        else:
            raise NotImplementedError

    def configure_scheduler(self) -> LambdaLR:
        if self.config.training.scheduler == "Constant":
            return ConstantLRSchedule(self.optimizer)
        elif self.config.training.scheduler == "WarmupConstant":
            return WarmupConstantSchedule(self.optimizer, self.config.training.warmup_steps)
        elif self.config.training.scheduler == "WarmupLinear":
            return WarmupLinearSchedule(self.optimizer, self.config.training.warmup_steps, self.__total_num_iterations)
        elif self.config.training.scheduler == "WarmupCosine":
            if self.config.traing.warmup_cosine_cycle is None:
                cycles = 0.5
            else:
                cycles = self.config.traing.warmup_cosine_cycle
            return WarmupCosineSchedule(
                self.optimizer, self.config.training.warmup_steps, self.__total_num_iterations, cycles=cycles
            )
        elif self.config.training.scheduler == "WarmupCosineWithHardRestartsSchedule":
            if self.config.traing.warmup_cosine_cycle is None:
                cycles = 0.5
            else:
                cycles = self.config.traing.warmup_cosine_cycle
            return WarmupCosineWithHardRestartsSchedule(
                self.optimizer, self.config.training.warmup_steps, self.__total_num_iterations, cycles=cycles
            )
        else:
            logger.error("Write your own version of `configure_scheduler`!")
            raise NotImplementedError

    def configure_model(self) -> nn.Module:
        logger.error("Please define your model first!")
        raise NotImplementedError

    def __save_checkpoint(self) -> None:
        states = self.state_dict()
        self.checkpointer.save_checkpoint("iter_" + str(self.__global_count), states)

    def __restore_checkpoint(self):
        logger.info("Try to restore the latest checkpoint")
        states = self.checkpointer.restore_latest_checkpoint()
        return states

    def state_dict(self):
        # TODO: add error handling for saving and loading rng states and amp states
        # which can simply be loading the first device's rng states

        if self.config.training.num_gpus_per_node > 1:
            model_states = self.model.module.state_dict()
        else:
            model_states = self.model.state_dict()
        states = {
            "epoch": self.__epochs_trained,
            "iteration": self.__global_count,
            "iteration_in_epoch": self.__count_in_epoch,
            "model_states": model_states,
            "optimizer_states": self.optimizer.state_dict(),
            "scheduler_states": self.scheduler.state_dict(),
            "checkpointer_states": self.checkpointer.state_dict(),
            "cpu_rng_states": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        # save amp states
        if self.config.training.fp16:
            states["amp"] = amp.state_dict()
        return states

    def load_state_dict(self, states: Dict[str, Any]):
        self.__epochs_trained = states["epoch"]
        self.__global_count = states["iteration"]
        self.__count_in_epoch = states["iteration_in_epoch"]
        self.model.load_state_dict(states["model_states"])
        self.optimizer.load_state_dict(states["optimizer_states"])
        self.scheduler.load_state_dict(states["scheduler_states"])
        self.checkpointer.load_state_dict(states["checkpointer_states"])
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

    def __set_random_seed(self, random_seed):
        # Reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def __log_iteration(self, batch_idx, results: Dict[str, torch.Tensor]):
        """
        Args:
           batch_idx: 
           results:
        """
        _loss = results['loss'].item() * self.config.training.num_gpus_per_node
        avg_loss = self.moving_average.update_key("loss", _loss)
        percent = 100. * self.__count_in_epoch / len(self.train_loader)
        logger.info(
            f"Train Epoch: [{self.__epochs_trained+1}/{self.__total_num_epochs}]"
            f" [{percent:7.4f}%]    Loss: {avg_loss:8.6f}"
        )

        if self.__tensorboard:
            self.__tensorboard.add_scalar("loss", _loss, self.__global_count + 1)


def set_random_port(self):
    """
    When running DDP NOT managed by SLURM, the ports might collide
    :return:
    """
    try:
        default_port = os.environ['MASTER_PORT']
    except Exception:
        import random
        default_port = random.randint(20000, 29000)
        os.environ['MASTER_PORT'] = str(default_port)
