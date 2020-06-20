import os
from omegaconf import OmegaConf
from typing import Any, Dict


class Singleton(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalFlyConfig(metaclass=Singleton):
    def __init__(self, config_path:str=None, config_file:str=None):
        self.initialized = False
        self.config = OmegaConf.create()
        self.old_cwd = os.getcwd()

        if config_path is not None:
            self.initialize(config_path, config_file)

    def initialize(self, config_path:str, config_file: str=None):
        if self.initialized:
            raise ValueError("FlyConfig is already initialized!")

        # Search config file
        if config_file is None:
            if os.path.exists(os.path.join(config_path, "config.yaml")):
                config_file = "config.yaml"
            elif os.path.exists(os.path.join(config_path, "config.yml")):
                config_file = "config.yml"
            else:
                raise ValueError("Cannot find config.yml. Please specify `config_file`")

        self.initialized = True
        
    def is_initialized(self) -> bool:
        return self.initialized

    def clear(self) -> None:
        self.initialized = False
        self.config = OmegaConf.create()