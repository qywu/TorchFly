import os
import ray
import logging
import hydra
from hydra.utils import get_original_cwd

import numpy as np
import torch
from torchvision import datasets, transforms

from torchfly.training.trainer import Trainer

from model import get_model
from dataloader import get_data_loader

logger = logging.getLogger(__name__)


def train_loader_fn(config):
    return get_data_loader(config)


@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    # set data loader
    val_loader = get_data_loader(config, evaluate=True)
    model = get_model()
    trainer = Trainer(config=config, model=model, validation_loader=None, train_loader_fn=train_loader_fn)
    trainer.train()


if __name__ == "__main__":
    main()
