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
    total_num_sectors = 128
    sector_size = total_num_sectors // config.training.num_gpus_per_node

    data = []
    try:
        rank = torch.distributed.get_rank()
    except:
        rank = 0
        
    for i in range(sector_size):
        filename = f"/home/wuqy1203/SIMPLE_DATA/{i + rank * sector_size}.pkl"
        data.extend(torch.load(filename))
        logger.info(f"{i + rank * sector_size}.pkl loaded")

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
