from typing import Any, Dict, List
import os
import sys
import copy
import time
import shutil
import logging
import logging.config
from omegaconf import OmegaConf, DictConfig
import argparse
import re
import torch

import torchfly.utils.distributed as distributed

logger = logging.getLogger(__name__)


class Singleton(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FlyLogger(metaclass=Singleton):
    """
    FlyLogger sets up the logger and output directory. It is a Singleton class that should be initialized once.

    """

    def __init__(self, config: OmegaConf, logging: bool = True, chdir: bool = True, overwrite: bool = False, resume: bool = False):
        """
        Initialize FlyLogger

        Args:
            config (OmegaConf): loaded by FlyConfig
            logging (bool): whether to setup the logger
            chdir (bool): whether to setup a new directory as in config
            overwrite (bool): whether to overwrite the existing logging folder, if not a new directory is created
            resume (bool): whether the training will be resumed
        """
        self.config = config
        self.logging = logging
        self.chdir = chdir
        self.overwrite = overwrite
        self.resume = resume
        # self.set_new_wd = not self.overwrite and not self.resume
        self.initialized = False

        if torch.distributed.is_initialized():
            self.rank = distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", 0))

        if self.overwrite and self.resume:
            raise ValueError("You cannot set `overwrite` and `resume` to True at the same time!")

        # self.initialize()
        logger.warning("Remember to use `with` statement to initialize !")

    def initialize(self):
        if self.initialized:
            raise ValueError("FlyLogger is already initialized! Please use `.clear()` before initialize it again.")

        # save original working directory
        owd = os.getcwd()
        self.config.flyconfig.runtime.owd = owd
        cwd = self.config.flyconfig.run.dir
        self.config.flyconfig.runtime.cwd = os.path.abspath(cwd)
        save_config_dir = self.config.flyconfig.save_config_dir

        if os.path.exists(cwd) and os.path.samefile(owd, cwd):
            raise ValueError("Please sepcify a sub-directory for `flyconfig.run.dir`")

        if not os.path.exists(cwd):
            # if cwd does not exist, directly create it
            if self.rank == 0:
                os.makedirs(cwd)
                self.save_config(os.path.join(cwd, save_config_dir))
        elif self.overwrite:
            # remove cwd and create a new one
            if self.rank == 0:
                logger.warning("Overwriting the current working directory!")
                shutil.rmtree(cwd)
                os.makedirs(cwd)
                self.save_config(os.path.join(cwd, save_config_dir))
        elif self.resume:
            # if resume, then do nothing
            pass
        else:
            # determine the current working directory name 
            count = 1
            copy_dir = os.path.abspath(cwd)
            copy_dir = copy_dir + "_copy_"
            while os.path.exists(copy_dir + str(count)):
                count += 1
            copy_dir = copy_dir + str(count)
            # change the working directory to the new one
            cwd = copy_dir
            self.config.flyconfig.runtime.cwd = cwd

            if self.rank == 0:
                os.makedirs(copy_dir)
                self.save_config(os.path.join(cwd, save_config_dir))

        distributed.barrier()
        
        # change the directory as in config
        if self.chdir:
            os.chdir(self.config.flyconfig.run.dir)
            self.config.flyconfig.runtime.cwd = os.getcwd()

        # configure logging, FlyLogger should only configure rank 0
        # other ranks should use their own logger
        if self.rank == 0 and self.logging:
            logging.config.dictConfig(OmegaConf.to_container(self.config.flyconfig.logging))
            logger.info("FlyLogger is initialized!")
            if self.chdir:
                logger.info(f"Working directory is changed to {os.getcwd()}")
        elif self.rank != 0 and self.logging:
            # for other ranks, we initialize a debug level logger
            logging.basicConfig(format=f'[%(asctime)s][%(name)s][%(levelname)s][RANK {self.rank}] - %(message)s',
                                level=logging.DEBUG)

        self.initialized = True

    def save_config(self, save_config_dir):
        config_path = self.config.flyconfig.runtime.config_path
        cwd_config_dirpath = os.path.join(self.config.flyconfig.runtime.owd, os.path.dirname(config_path))
        save_config_dir = os.path.join(save_config_dir, "saved_config")
        #  os.makedirs(save_config_dir, exist_ok=True)

        if not os.path.exists(save_config_dir):
            shutil.copytree(cwd_config_dirpath, save_config_dir)
            final_config_path = os.path.join(save_config_dir, "all_config.yml")
            with open(final_config_path, "w") as f:
                OmegaConf.save(self.config, f)

    def __enter__(self):
        if self.initialized:
            logger.warning("FlyLogger is initialized with `__init__`!")
        else:
            self.initialize()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def is_initialized(self) -> bool:
        return self.initialized

    def clear(self) -> None:
        "A duplicate of self.close"
        self.initialized = False
        # change back to the original working directory
        os.chdir(self.config.flyconfig.runtime.owd)

    def close(self) -> None:
        self.initialized = False
        # change back to the original working directory
        os.chdir(self.config.flyconfig.runtime.owd)