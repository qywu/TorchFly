import ray
import torch
import numpy as np
import logging
from collections import OrderedDict
from typing import Any, Dict

logger = logging.getLogger(__name__)


def copy_cpu_state_dict(states: Any) -> Dict[str, Any]:
    # need a new dict
    result_states = OrderedDict()

    if isinstance(states, dict):
        # recursion
        for k in states:
            result_states[k] = copy_cpu_state_dict(states[k])
    elif isinstance(states, list):
        result_states = [copy_cpu_state_dict(item) for item in states]
    elif isinstance(states, torch.Tensor):
        result_states = states.cpu().numpy()
    elif isinstance(states, (int, float, str, tuple, type(None))):
        result_states = states
    else:
        result_states = states
        logging.warn(f"`copy_cpu_state_dict` cannot parse {type(states)}")
        # print(f"`copy_cpu_state_dict` cannot parse {type(states)}")
    return result_states


def _convert_numpy_to_torch_state_dict(states: Dict[str, Any]) -> Dict[str, Any]:
    # No need to copy states
    if isinstance(states, dict):
        # recursion
        for k in states:
            states[k] = _convert_numpy_to_torch_state_dict(states[k])
    elif isinstance(states, list):
        states = [_convert_numpy_to_torch_state_dict(item) for item in states]
    elif isinstance(states, np.ndarray):
        states = torch.from_numpy(states)
    elif isinstance(states, (int, float, str, tuple, type(None))):
        states = states
    else:
        states = states
        logging.warn(f"`_convert_numpy_to_torch_state_dict` cannot parse {type(states)}")
    return states


@ray.remote
def _async_save(states, filename):
    states = _convert_numpy_to_torch_state_dict(states)
    torch.save(states, filename)
    return 0


def async_save(model_states: OrderedDict, filename):
    model_states = copy_cpu_state_dict(model_states)
    ray_obj = _async_save.remote(model_states, filename)
    return ray_obj


def check_async_status(ray_obj):
    return ray.get(ray_obj)