import os
import torch.utils.data._utils as _utils
from typing import Any, Iterator


class DataProcessor:
    """
    An abstract class for processing items.
    This is the one which needs to be implemented
    """
    def __init__(self):
        self.rank = os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else 0

    def __call__(self, item):
        return self.process(item)

    def process(self, item: Any) -> Iterator:
        raise NotImplementedError("Please implement this function before use")

    def collate_fn(self, batch):
        return _utils.collate.default_collate(batch)