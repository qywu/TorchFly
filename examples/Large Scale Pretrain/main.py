import os
import ray
import logging
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from torchvision import datasets, transforms
from torchfly.training.trainer import Trainer
from torchfly.text import IterableDataLoader

from text_dataset import FullDocDataset, FullDocCollate
from model import PretrainModel

logger = logging.getLogger(__name__)

def train_loader_fn(config):
    data = torch.load("/home/wuqy1203/Desktop/Projects/TorchFly/examples/Large Scale Pretrain/data/0.pkl")
    # roberta-base.sep_token_id = 2
    def _dataset_init_fn(index):
        return FullDocDataset(data, max_seq_len=510, sep_token_id=2)

    collate_fn = FullDocCollate(config)

    return IterableDataLoader(dataset_init_fn=_dataset_init_fn, batch_size=config.training.batch_size, collate_fn=collate_fn)


@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    # set data loader
    model = PretrainModel(config)
    trainer = Trainer(config=config, train_loader_fn=train_loader_fn, model=model)
    trainer.train()


if __name__ == "__main__":
    main()
