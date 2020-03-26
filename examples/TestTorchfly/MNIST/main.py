import os
import ray
import logging
import hydra
from hydra.utils import get_original_cwd

import numpy as np
import torch
from torchvision import datasets, transforms

# from torchfly_dev.training.trainer import Trainer
from torchfly.training.trainer import Trainer

from model import CNNNet
from dataloader import get_data_loader

logger = logging.getLogger(__name__)


@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    # set data loader
    train_loader, val_loader = get_data_loader(config)
    model = CNNNet(config)
    trainer = Trainer(config=config, model=model, train_loader=train_loader, validation_loader=None)
    trainer.train()

if __name__ == "__main__":
    main()
