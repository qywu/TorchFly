import ray
import logging
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from typing import Callable

logger = logging.getLogger(__name__)


class IterableDataLoader:
    def __init__(
        self,
        dataset: Dataset = None,
        dataset_init_fn: Callable = None,
        batch_size: int = 1,
        collate_fn: Callable = None,
        **kwargs
    ):
        self.dataset = dataset
        self.dataset_init_fn = dataset_init_fn
        self.batch_size = batch_size

        if collate_fn:
            self.collate_fn = collate_fn
        else:
            self.collate_fn = default_collate

        # we independently initialize each iterator
        if dataset_init_fn is not None:
            self.datasets = [dataset_init_fn(index=i, **kwargs) for i in range(batch_size)]

        # if not ray.is_initialized():
        #     ray.init()
        #     logger.warning("Ray is initialized!")

    def __iter__(self):
        #iterators = ray.util.iter.from_iterators(self.datasets).gather_sync()
        iterators = [iter(item) for item in self.datasets]

        while True:
            try:
                # TODO: extend it with Ray
                # batch = next(iterators)
                batch = [next(iterators[i]) for i in range(self.batch_size)]
                yield self.collate_fn(batch)
            except StopIteration:
                break

    def __len__(self):
        return len(self.datasets[0])
