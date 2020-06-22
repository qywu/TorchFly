import os
import sys
import signal
import logging
import tqdm
import datetime
import torch
import torch.utils.data._utils as _utils
from torch.utils.data import IterableDataset
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process, Queue, Event
import numpy as np
import random
import atexit
from omegaconf import OmegaConf
import pyarrow.plasma as plasma
from pyarrow.plasma import PlasmaObjectExists
import queue

from typing import Any, Iterator

from .data_processor import DataProcessor
from .plasma import GlobalPlasmaManager

logger = logging.getLogger(__name__)

# pylint:disable=no-member

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


class _DataMover(Process):
    def __init__(
        self, config: OmegaConf, local_rank: int, random_seed: int, dataset: IterableDataset, in_queue: Queue,
        done_event: Event
    ):
        """
        Args:
            config:
            local_rank:
            dataset:
        """
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.dataset = dataset

        # hidden variables
        self._random_seed = random_seed
        self._done_event = done_event
        self._in_queue = in_queue

        # plasma client cannot be pickled
        # therefore, we must initialize it in `self.run`
        self._plasma_client = None

    def run(self):
        torch.set_num_threads(1)
        # DO NOT use singleton pattern for multiprocessing!
        self._plasma_client = plasma.connect(f"/tmp/torchfly/plasma/{self.config.plasma.plasma_store_name}/plasma.sock")
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
        self, config: OmegaConf, local_rank: int, random_seed: int, processor: DataProcessor, in_queue: Queue,
        out_queue: Queue, done_event: Event
    ):
        """
        Args:
            config:
            local_rank:
            random_seed:
            processor: a function that processes data
        """
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.processor = processor

        # hidden variables
        self._random_seed = random_seed
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._done_event = done_event
        self._old_obj_buffer = []

        # plasma client cannot be pickled
        # therefore, we must initialize it in `self.run`
        self._plasma_client = None

    def run(self):
        torch.set_num_threads(1)
        self._plasma_client = plasma.connect(f"/tmp/torchfly/plasma/{self.config.plasma.plasma_store_name}/plasma.sock")
        set_random_seed(self._random_seed)

        # need to make sure each processor has different local rank
        if self.processor:
            self.processor.rank = self.local_rank

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
                #continue

            if self._done_event.is_set():
                break

            if len(self._old_obj_buffer) >= 512:
                self._plasma_client.delete(self._old_obj_buffer)
                self._old_obj_buffer = []


class FlyDataLoader:
    def __init__(
        self,
        config: OmegaConf,
        dataset: IterableDataset,
        processor: DataProcessor = None,
        collate_fn=None,
        post_process_fn=None,
    ):
        """
        Args:
        """
        self.config = config
        self.dataset = dataset
        self.processor = processor
        self.post_process_fn = post_process_fn
        self.rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0

        self.batch_size = config.dataloader.batch_size
        self.sync_output = config.dataloader.sync_output

        self.timeout = config.dataloader.timeout
        self.drop_last = config.dataloader.drop_last

        # Set the number of movers
        self.num_movers = config.dataloader.num_movers
        # Set the number of workers
        if self.sync_output:
            self.num_workers = self.batch_size

            if self.batch_size != config.dataloader.num_workers:
                logger.warn(f"SyncOutput On! Overwriting number of workers to be {self.batch_size}")
        else:
            self.num_workers = config.dataloader.num_workers
        # Set number of output queues
        self._num_out_queues = self.batch_size if self.sync_output else 1

        self._worker_pids_set = False
        self._done_event = Event()

        # Set processor collate function
        if processor:
            self._collate_fn = collate_fn if collate_fn else processor.collate_fn
        else:
            self._collate_fn = _utils.collate.default_collate

        # Set number of input queues
        if self.num_movers > 0:
            self._in_queue_size = config.dataloader.in_queue_size_multiplier * self.num_workers
        else:
            self._in_queue_size = len(dataset)

        self._out_queue_size = config.dataloader.out_queue_size_multiplier * self.batch_size

        self._internal_counter = 0
        self._worker_pids_set = True
        self._old_obj_buffer = []
        self.object_buffer_size = config.dataloader.object_buffer_size

        # Init Plasma
        self._plasma_client = GlobalPlasmaManager(self.config.plasma).client
        self._random_seed = config.random_seed
        set_random_seed(self._random_seed)

        _utils.signal_handling._set_SIGCHLD_handler()
        # signal.signal(signal.SIGINT, self.keyboardInterruptHandler)
        atexit.register(self.__del__)
        

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # Prepare Queues
        self._in_queue = Queue(maxsize=self._in_queue_size)
        self._out_queues = [Queue(maxsize=self._out_queue_size) for _ in range(self._num_out_queues)]
        self.empty_queues = set()

        # Start Working
        self.start_movers(self.dataset)
        self.start_workers(self.processor)

        return self

    def __next__(self):
        # Get a Batch
        if self.sync_output:
            worker_indices, batch = self._sync_get_batch()
        else:
            worker_indices, batch = self._no_sync_get_batch()

        if len(self._old_obj_buffer) >= self.object_buffer_size:
            self._plasma_client.delete(self._old_obj_buffer)
            self._old_obj_buffer = []

        if self.post_process_fn:
            return self.post_process_fn(worker_indices, batch)
        else:
            return batch

    def _sync_get_batch(self):
        batch = []
        worker_indices = []

        for idx in range(self.batch_size):
            if idx in self.empty_queues:
                continue
            try:
                obj_id = self._out_queues[idx].get(timeout=self.timeout)
                product = self._plasma_client.get(obj_id)
                batch.append(product)
                worker_indices.append(idx)
                self._old_obj_buffer.append(obj_id)
            except queue.Empty:
                if self.drop_last:
                    batch = []
                    break
                self.empty_queues.add(idx)
                continue

        if len(batch) == 0:
            # End all multiprocessing parts
            self._in_queue.close()
            [out_queue.close() for out_queue in self._out_queues]
            self.kill_movers()
            self.kill_workers()
            self._internal_counter += 300
            raise StopIteration

        return worker_indices, self._collate_fn(batch)

    def _no_sync_get_batch(self):
        batch = []
        worker_indices = []

        for idx in range(self.batch_size):
            try:
                obj_id = self._out_queues[0].get(timeout=self.timeout)
                product = self._plasma_client.get(obj_id)
                batch.append(product)
                worker_indices.append(idx)
                self._old_obj_buffer.append(obj_id)
            except queue.Empty:
                if self.drop_last:
                    batch = []
                break

        if len(batch) == 0:
            # End all multiprocessing parts
            self._in_queue.close()
            [out_queue.close() for out_queue in self._out_queues]
            self.kill_movers()
            self.kill_workers()
            self._internal_counter += 300
            raise StopIteration

        return worker_indices, self._collate_fn(batch)

    def start_movers(self, dataset: IterableDataset):
        # Start Movers
        # mover need to have a different random seed than all other processes
        if self.num_movers > 0:
            self._movers = [
                _DataMover(
                    config=self.config,
                    local_rank=local_rank,
                    random_seed=local_rank + self._internal_counter + self._random_seed + self.rank * self.num_movers +
                    900000,
                    dataset=dataset,
                    in_queue=self._in_queue,
                    done_event=self._done_event,
                ) for local_rank in range(self.num_movers)
            ]
            for mover in self._movers:
                mover.daemon = True
                mover.start()

        else:
            self._movers = [
                _DataMover(
                    config=self.config,
                    local_rank=0,
                    random_seed=self._random_seed + 10000,
                    dataset=dataset,
                    in_queue=self._in_queue,
                    done_event=self._done_event
                )
            ]
            self._movers[0].run()

        # pause while queue is not empty
        while self._in_queue.empty():
            pass

    def start_workers(self, processor: DataProcessor):
        # All the workers need to have different random seeds
        # Start Workers
        self._workers = []

        for local_rank in range(self.num_workers):
            if self.sync_output:
                out_queue = self._out_queues[local_rank]
            else:
                out_queue = self._out_queues[0]

            worker = _DataWorker(
                config=self.config,
                local_rank=local_rank + self.rank * self.num_workers,
                random_seed=local_rank + self._internal_counter + self._random_seed + self.rank * self.num_workers,
                processor=processor,
                in_queue=self._in_queue,
                out_queue=out_queue,
                done_event=self._done_event
            )
            self._workers.append(worker)

        for worker in self._workers:
            worker.daemon = True
            worker.start()

        # wait until there is something in the queue
        while any([q.empty() for q in self._out_queues]):
            pass

    def kill_workers(self):
        for w in self._workers:
            if w.is_alive():
                w.kill()

    def kill_movers(self):
        if self.num_movers > 0:
            for m in self._movers:
                if m.is_alive():
                    m.kill()

    def __del__(self):
        "Destructor handle the cleaning of processes"
        self.kill_movers()
        self.kill_workers()

    def keyboardInterruptHandler(self, signal, frame):
        logger.info("Keyboard Terminated!")
        self._in_queue.close()
        [out_queue.close() for out_queue in self._out_queues]
        self.kill_movers()
        self.kill_workers()
        self._internal_counter += 10000
        raise KeyboardInterrupt