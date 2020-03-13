import torch
import numpy as np
import random

from .common import async_save, check_async_status


def init():
    raise NotImplementedError


__all__ = ["async_save", "check_async_status", "init"]