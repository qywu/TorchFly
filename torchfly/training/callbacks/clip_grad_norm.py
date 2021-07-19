import os
import sys
import time
import torch
import logging
from omegaconf import DictConfig
from typing import Any, Dict

from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)
Trainer = Any


@Callback.register("gradient_clip_norm")
class GradientClipNorm(Callback):
    """
    Clip the gradient based on its norm
    """
    @handle_event(Events.STEP_BEGIN, priority=-100)
    def gradient_clip_norm(self, trainer: Trainer):
        raise NotImplementedError