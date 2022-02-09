from typing import Any, List, Dict, Iterator, Callable
import tqdm
import torch
import torch.nn as nn
import numpy as np

# import apex
# from apex.parallel import DistributedDataParallel, Reducer
# from torch.nn.parallel import DistributedDataParallel

from torchfly.utilities import move_to_device
from torchfly.metrics import CategoricalAccuracy, Average, MovingAverage, Speed
from torchfly.training.schedulers import (
    ConstantLRSchedule,
    WarmupConstantSchedule,
    WarmupCosineSchedule,
    WarmupLinearSchedule,
    WarmupCosineWithHardRestartsSchedule,
)

import logging

logger = logging.getLogger(__name__)


class FlyModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.is_training = True

    def set_trainer(self, trainer):
        """
        Enable FlyModel to access trainer loop information
        """
        self.trainer = trainer

    def predict_step(self, batch_idx, dataloder_idx=0, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_metrics(self, reset):
        return {}

    def get_optimizer_parameters(self, config=None):
        """
        This function is used to set parameters with different weight decays
        """
        try:
            weight_decay = config.optimization.weight_decay
        except:
            weight_decay = 0.01

        # default groups
        no_decay = ["bias", "Norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self, config, total_num_update_steps) -> [List, List]:
        optimizer_grouped_parameters = self.get_optimizer_parameters(config)
        lr = config.optimization.learning_rate
        optimizer_name = config.optimization.optimizer_name
        max_gradient_norm = config.optimization.max_gradient_norm
        betas = (
            config.optimization.betas
            if config.optimization.get("betas")
            else (0.9, 0.999)
        )

        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=lr, betas=betas
            )
        elif optimizer_name == "Adafactor":
            raise NotImplementedError
        elif optimizer_name == "Adadelta":
            optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=lr)
        else:
            raise NotImplementedError(
                f"{optimizer_name} is not implemented! Override FlyModel's configure optimizer to continue!"
            )

        scheduler_name = config.scheduler.scheduler_name
        warmup_steps = config.scheduler.get("warmup_steps", None)
        warmup_cycle = config.scheduler.get("warmup_cosine_cycle", None)

        if scheduler_name == "Constant":
            scheduler = ConstantLRSchedule(optimizer)
        elif scheduler_name == "WarmupConstant":
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps)
        elif scheduler_name == "WarmupLinear":
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps, total_num_update_steps
            )
        elif scheduler_name == "WarmupCosine":
            if warmup_cycle is None:
                warmup_cycle = 0.5
            scheduler = WarmupCosineSchedule(
                optimizer, warmup_steps, total_num_update_steps, cycles=warmup_cycle
            )
        elif scheduler_name == "WarmupCosineWithHardRestartsSchedule":
            if warmup_cycle is None:
                warmup_cycle = 0.5

            scheduler = WarmupCosineWithHardRestartsSchedule(
                optimizer, warmup_steps, total_num_update_steps, cycles=warmup_cycle
            )
        else:
            logger.error("Write your own version of `configure_scheduler`!")
            raise NotImplementedError

        setattr(self, "get_last_lr", scheduler.get_last_lr)

        return [optimizer], [scheduler]

    def traininging_step(self, train_batch, batch_idx):
        return self.forward(train_batch)

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self.predict_step(val_batch)

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        return self.predict_step(test_batch)

    def validation_loop(self, dataloader):
        # No gradient is needed for validation
        self.eval()
        self.reset_evaluation_metrics()
        with torch.no_grad():
            pbar = tqdm.tqdm(dataloader)
            pbar.mininterval = 2.0
            for batch_idx, batch in enumerate(pbar):
                # send to cuda device
                batch = move_to_device(batch, self.device)
                self.validation_step(batch, batch_idx)

    def test_loop(self, dataloader):
        self.eval()
        self.reset_evaluation_metrics()
        # No gradient is needed for validation
        with torch.no_grad():
            pbar = tqdm.tqdm(dataloader)
            pbar.mininterval = 2.0
            for batch_idx, batch in enumerate(pbar):
                # send to cuda device
                batch = move_to_device(batch, self.device)
                self.test_step(batch, batch_idx)

    def get_last_lr(self):
        raise NotImplementedError(
            "Please hook this function to the `scheduler.get_last_lr`!"
        )

    def get_training_metrics(self) -> Dict[str, str]:
        raise NotImplementedError

    def get_evaluation_metrics(self) -> Dict[str, str]:
        raise NotImplementedError

    def reset_training_metrics(self):
        for metric in self.training_metrics.values():
            metric.reset()

    def reset_evaluation_metrics(self):
        for metric in self.evaluation_metrics.values():
            metric.reset()

    def configure_metrics(self):
        self.training_metrics = {"loss": MovingAverage()}
        self.evaluation_metrics = {"loss": Average()}

    # def to(self, device, non_blocking=False):
    #     pass

    # def to_distributed(self):
    #     for key, _ in self.named_children():
    #         model = getattr(self, key)
    #         ddp_model = DistributedDataParallel(model, delay_allreduce=True)
    #         setattr(self, key, ddp_model)

    # def to_single(self):
    #     for key, _ in self.named_children():
    #         model = getattr(self, key).module
    #         setattr(self, key, model)
