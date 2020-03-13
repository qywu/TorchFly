import os
import torch
import hydra
import argparse
import copy
# from omegaconf import DictConfig
# from simple_trainer import SimpleTrainer

g_config = None

@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    # setup
    global g_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    config.training.local_rank = args.local_rank
    g_config = config

    # define trainer
    



if __name__ == "__main__":
    main()
    breakpoint()