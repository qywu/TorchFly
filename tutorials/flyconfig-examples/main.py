import os
import sys
from typing import Any, Dict, List
import logging
import copy
from omegaconf import OmegaConf

from torchfly.flyconfig.flyconfig import FlyConfig

config = FlyConfig.load(config_path=None)
FlyConfig.print(config)