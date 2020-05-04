import os
import ray
import logging
import hydra
from hydra.utils import get_original_cwd

import numpy as np
import torch
import random
from torchvision import datasets, transforms

# from torchfly_dev.training.trainer import Trainer
from torchfly.training.trainer import TrainerLoop
from torchfly.common import set_random_seed

from model import CNNNet
from dataloader import DataHandler

logger = logging.getLogger(__name__)

# this line must be called before everything else


@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    set_random_seed(config.training.random_seed)
    # set data loader
    data_handler = DataHandler(config)

    model = CNNNet(config)
    trainer = TrainerLoop(
        config=config,
        model=model,
        train_dataloader_fn=data_handler.train_loader,
        valid_dataloader_fn=data_handler.valid_loader
    )
    trainer.train()


if __name__ == "__main__":
    main()
