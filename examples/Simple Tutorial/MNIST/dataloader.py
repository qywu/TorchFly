import os
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cwd():
    try:
        return get_original_cwd()
    except AttributeError:
        return os.getcwd()

class CycleDataloader(torch.utils.data.DataLoader):
    """
    A warpper to DataLoader that the `__next__` method will never end. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = None

    def __iter__(self):
        for batch in super().__iter__():
            batch = {
                "input": batch[0],
                "target": batch[1]
            }
            yield batch

class DataHandler:
    def __init__(self, config):
        self.config = config

    def train_loader(self, config):
        kwargs = {'num_workers': 0, 'pin_memory': True}
        train_loader = CycleDataloader(
            datasets.MNIST(os.path.join(get_cwd(), 'data'), train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=config.training.batch_size, shuffle=True, **kwargs)
        return train_loader

    def valid_loader(self, config):
        kwargs = {'num_workers': 0, 'pin_memory': True}
        test_loader = CycleDataloader(
        datasets.MNIST(os.path.join(get_cwd(), 'data'), train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=config.training.batch_size, shuffle=True, **kwargs)

        return test_loader