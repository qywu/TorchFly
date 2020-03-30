import os
import tqdm
import datetime
import pyarrow.plasma as plasma
from pyarrow._plasma import PlasmaObjectExists
import torch
import torch.utils.data._utils as _utils
from torch.utils.data import IterableDataset
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process, Queue, Event
import numpy as np
import random
import queue
import atexit
import logging
from typing import Any, Iterator

from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


class _DataMover(Process):
    def __init__(
        self, local_rank: int, random_seed: int, dataset: IterableDataset, in_queue: Queue, done_event: Event,
        plasma_store_address: str
    ):
        super().__init__()
        self.local_rank = local_rank
        self.dataset = dataset
        self._random_seed = random_seed
        self._done_event = done_event
        self._in_queue = in_queue
        self._plasma_store_address = plasma_store_address

        # plasma client cannot be pickled
        # therefore, we must initialize it in `self.run`
        self._plasma_client = None

    def run(self):
        self._plasma_client = plasma.connect(self._plasma_store_address)
        set_random_seed(self.local_rank + self._random_seed)

        for item in self.dataset:
            try:
                obj_id = self._plasma_client.put(item)
                self._in_queue.put(obj_id)
            except PlasmaObjectExists:
                logger.error(f"{datetime.datetime.now()} mover repeat")
                continue

            if self._done_event.is_set():
                break


class _DataWorker(Process):
    def __init__(
        self, local_rank: int, random_seed: int, processor: DataProcessor, in_queue: Queue, out_queue: Queue,
        done_event: Event, plasma_store_address: str
    ):
        super().__init__()
        self.local_rank = local_rank
        self.processor = processor
        self._random_seed = random_seed
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._done_event = done_event
        self._plasma_store_address = plasma_store_address
        self._old_obj_buffer = []

        # plasma client cannot be pickled
        # therefore, we must initialize it in `self.run`
        self._plasma_client = None

    def run(self):
        torch.set_num_threads(1)
        self._plasma_client = plasma.connect(self._plasma_store_address)
        set_random_seed(self.local_rank + self._random_seed)

        if self.processor:
            self.processor.set_rank(self.local_rank)

        while True:
            obj_id = self._in_queue.get()
            item = self._plasma_client.get(obj_id)
            self._old_obj_buffer.append(obj_id)

            try:
                # Call Processor to produce an iterator
                if self.processor:
                    for product in self.processor.process(item):
                        obj_id = self._plasma_client.put(product)
                        self._out_queue.put(obj_id)
                else:
                    obj_id = self._plasma_client.put(item)
                    self._out_queue.put(obj_id)
            except PlasmaObjectExists:
                logger.error(f"{datetime.datetime.now()} worker repeat")
                continue

            if self._done_event.is_set():
                break

            if len(self._old_obj_buffer) >= 512:
                self._plasma_client.delete(self._old_obj_buffer)
                self._old_obj_buffer = []


class DataDispatcher:
    def __init__(
        self,
        dataset: IterableDataset,
        plasma_store_address: str,
        processor: DataProcessor = None,
        batch_size: int = 1,
        in_queue_size_multiplier: int = 20,
        out_queue_size_multiplier: int = 20,
        sorted_output: bool = True,
        num_movers: int = 1,
        num_workers: int = 1,
        random_seed: int = 123,
        timeout: int = 10,
        collate_fn=None,
        local_rank: int = 0,
    ):
        """
        Args:
            in_queue_size_multiplier: how many times the in queue size is bigger than the number of workers
            sorted_output: this will cause DataDispatcher to have the same number of out_queues as the batch size 
        """
        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        self.batch_size = batch_size
        self.sorted_output = sorted_output
        self.dispatcher_rank = local_rank
        self._random_seed = random_seed
        self.timeout = timeout
        self.num_movers = num_movers
        self.num_workers = batch_size if sorted_output else num_workers
        self._num_out_queues = batch_size if sorted_output else 1
        self._worker_pids_set = False
        self._done_event = Event()
        if processor:
            self._collate_fn = collate_fn if collate_fn else processor.collate_fn
        else:
            self._collate_fn = _utils.collate.default_collate
        self._plasma_store_address = plasma_store_address

        set_random_seed(self._random_seed + self.dispatcher_rank)
        self._plasma_client = plasma.connect(plasma_store_address)

        self._in_queue_size = in_queue_size_multiplier * num_workers if self.num_movers > 0 else len(dataset)
        self._in_queue = Queue(maxsize=self._in_queue_size)

        self._out_queue_size = out_queue_size_multiplier * batch_size
        self._out_queues = [Queue(maxsize=self._out_queue_size) for _ in range(self._num_out_queues)]

        self.start_movers(dataset)
        self.start_workers(processor)

        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._movers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._old_obj_buffer = []

        atexit.register(self.__del__)

    def __iter__(self):
        return self

    def __next__(self):
        batch = []

        try:
            for idx in range(self.batch_size):
                if self.sorted_output:
                    obj_id = self._out_queues[idx].get(timeout=self.timeout)
                else:
                    obj_id = self._out_queues[0].get(timeout=self.timeout)
                product = self._plasma_client.get(obj_id)
                batch.append(product)
                self._old_obj_buffer.append(obj_id)
        except queue.Empty:
            raise StopIteration

        if len(self._old_obj_buffer) >= 512:
            self._plasma_client.delete(self._old_obj_buffer)
            self._old_obj_buffer = []

        return self._collate_fn(batch)

    def start_movers(self, dataset: IterableDataset):
        # Start Movers
        # mover need to have a different random seed than all other processes
        if self.num_movers > 0:
            self._movers = [
                _DataMover(
                    local_rank=i,
                    random_seed=self._random_seed + self.dispatcher_rank * self.num_movers + 10000,
                    dataset=dataset,
                    in_queue=self._in_queue,
                    done_event=self._done_event,
                    plasma_store_address=self._plasma_store_address
                ) for i in range(self.num_movers)
            ]
            for mover in self._movers:
                mover.daemon = True
                mover.start()

        else:
            self._movers = [
                _DataMover(
                    local_rank=-1,
                    random_seed=self._random_seed + 10000,
                    dataset=dataset,
                    in_queue=self._in_queue,
                    done_event=self._done_event,
                    plasma_store_address=self._plasma_store_address
                )
            ]
            self._movers[0].run()

        while self._in_queue.empty():
            pass

    def start_workers(self, processor: DataProcessor):
        # All the workers need to have different random seeds
        # Start Workers
        self._workers = []

        for local_rank in range(self.num_workers):
            if self.sorted_output:
                out_queue = self._out_queues[local_rank]
            else:
                out_queue = self._out_queues[0]

            worker = _DataWorker(
                local_rank=local_rank + self.dispatcher_rank * self.num_workers,
                random_seed=self._random_seed,
                processor=processor,
                in_queue=self._in_queue,
                out_queue=out_queue,
                done_event=self._done_event,
                plasma_store_address=self._plasma_store_address,
            )
            self._workers.append(worker)

        for worker in self._workers:
            worker.daemon = True
            worker.start()

        # wait until there is something in the queue
        while any([q.empty() for q in self._out_queues]):
            pass

    def __del__(self):
        "Destructor handle the cleaning of processes"
        if self.num_movers > 0:
            for m in self._movers:
                if m.is_alive():
                    m.kill()

        for w in self._workers:
            if w.is_alive():
                w.kill()