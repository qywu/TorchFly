import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchfly.flylogger import FlyLogger
from torchfly.flyconfig import FlyConfig
from torchfly.training import Trainer
import torchfly.distributed as distributed
from torchfly.utilities import set_random_seed
from omegaconf import OmegaConf

from model import CNNFlyModel


class DataLoaderHelper:

    def __init__(self, config):
        self.config = config

    def train_loader_fn(self):
        kwargs = {
            'num_workers': 0,
            'pin_memory': True,
        }

        with distributed.mutex() as rank:

            dataset = datasets.MNIST(os.path.join(self.config.task.datadir, 'MNIST'),
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose(
                                         [transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))


        train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=distributed.get_world_size(),
                rank=distributed.get_rank()
            )

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, **kwargs)
        return dataloader

    def valid_loader_fn(self):
        kwargs = {
            'num_workers': 0,
            'pin_memory': True,
        }
        dataset = datasets.MNIST(os.path.join(self.config.task.datadir, 'MNIST'),
                                 train=False,
                                 download=True,
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))]))

        dataloader = DataLoader(dataset, batch_size=self.config.training.evaluation.batch_size, shuffle=False, **kwargs)

        return dataloader


def main():
    # we recommand adding this function before everything starts
    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    config = FlyConfig.load("config/config.yaml")
    set_random_seed(config.training.random_seed)

    data_helper = DataLoaderHelper(config)
    train_dataloader = data_helper.train_loader_fn()
    valid_dataloader = data_helper.valid_loader_fn()

    model = CNNFlyModel(config)
    model.configure_metrics()

    trainer = Trainer(config.training, model)

    with FlyLogger(config.flylogger) as flylogger:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f)    
        trainer.train(config.training, train_dataloader, valid_dataloader)



if __name__ == "__main__":
    main()
