import os
import ray
import json
import tqdm
import torch
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class SentenceSegmenter:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, doc_sentences) -> List[List[str]]:

        token_segments = []
        current_seq = []

        for count, sent in enumerate(doc_sentences):
            if count > 0:
                sent = " " + sent

            token_sent = self.tokenizer.tokenize(sent)

            if len(token_sent) > self.max_seq_length:
                # append last sequence
                token_segments.append(current_seq)

                for i in range(0, len(token_sent) - self.max_seq_length, self.max_seq_length):
                    token_segments.append(token_sent[i:i + self.max_seq_length])

                # assign the current seq the tail of token_sent
                current_seq = token_sent[i + self.max_seq_length:i + self.max_seq_length * 2]
                continue

            if (len(current_seq) + len(token_sent)) > self.max_seq_length:
                token_segments.append(current_seq)
                current_seq = token_sent
            else:
                current_seq = current_seq + token_sent

        if len(current_seq) > 0:
            token_segments.append(current_seq)

        # remove empty segment
        token_segments = [seg for seg in token_segments if seg]

        return token_segments


@ray.remote
def _process(segmenter, lines, rank):
    all_token_segments = []

    # progress bar
    if rank == 0:
        lines = tqdm.tqdm(lines)
    else:
        lines = lines

    for line in lines:
        example = json.loads(line)
        token_segments = segmenter(example["sents"])
        all_token_segments.append(token_segments)

    return all_token_segments


class SegmentDocLoader:
    def __init__(
        self, tokenizer, max_seq_length: int, corpus_path: str, cache_dir: str = "cache/cached_corpus_sectors"
    ):
        self.tokenizer = tokenizer
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.sent_segmenter = SentenceSegmenter(tokenizer, max_seq_length)

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if "OMP_NUM_THREADS" not in os.environ.keys():
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)

    def load_sector(self, sector_id):
        if self.cache_dir:

            cache_path = os.path.join(self.cache_dir, f"{sector_id}_cache.pkl")

            if os.path.exists(cache_path):
                try:
                    logger.info("Loadding Cache")
                    processed_docs = torch.load(cache_path)
                    logger.info("Finished Loading")
                    return processed_docs
                except:
                    logger.info("File Corrupted. Data will be re-processed")

        # processing data
        with open(os.path.join(self.corpus_path, str(sector_id) + ".jsonl"), "r") as f:
            data = f.readlines()

        processed_docs = []

        logger.info("Processing Data. Takes about 10 mins")

        # multi-processing
        ray_objs = []
        step_size = len(data) // int(os.environ["OMP_NUM_THREADS"])
        for i in range(0, len(data), step_size):
            ray_objs.append(_process.remote(self.sent_segmenter, data[i:i + step_size], i))

        for i in range(len(ray_objs)):
            processed_docs.extend(ray.get(ray_objs[i]))

        if self.cache_dir:
            logger.info("Saving Into Cache")
            torch.save(processed_docs, cache_path)
            logger.info("Finished Saving Into Cache")

        return processed_docs