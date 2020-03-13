import os
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
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

    def __next__(self):
        if self.iterator is None:
            self.iterator = self.__iter__()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = self.__iter__()
            batch = next(self.iterator)
        return batch

def get_data_loader(config):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = CycleDataloader(
        datasets.MNIST(os.path.join(get_cwd(), 'data'), train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=config.training.batch_size, shuffle=True, **kwargs)

    test_loader = CycleDataloader(
    datasets.MNIST(os.path.join(get_cwd(), 'data'), train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=config.training.batch_size, shuffle=True, **kwargs)


    return train_loader, test_loader