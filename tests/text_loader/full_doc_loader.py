import os
import ray
import json
import tqdm
import torch
import logging
from typing import List

logger = logging.getLogger(__name__)


@ray.remote
def _process(tokenizer, lines: List[str], rank: int):
    """
    Args:
        tokenizer: huggingface's style tokenizer
        lines: jsonl line. with the key "sents"
        rank: for distribtued process to identify
    Returns:
        all_tokens: processed tokens for the documents 
    """
    all_tokens = []

    # progress bar
    if rank == 0:
        lines = tqdm.tqdm(lines)
    else:
        lines = lines

    for line in lines:
        example = json.loads(line)
        text = " ".join(example["sents"])
        doc_tokens = tokenizer.tokenize(text)
        all_tokens.append(doc_tokens)

    return all_tokens


class FullDocLoader:
    def __init__(self, tokenizer, corpus_path: str, cache_dir: str = "cache/full_doc_sectors"):
        self.tokenizer = tokenizer
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if "OMP_NUM_THREADS" not in os.environ.keys():
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)
            self.num_threads = os.cpu_count() // 2
        else:
            self.num_threads = int(os.environ["OMP_NUM_THREADS"])
            
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

        print("Processing Data. Takes about 10 mins")

        # multi-processing
        ray_objs = []
        step_size = len(data) // self.num_threads
        for i in range(0, len(data), step_size):
            ray_objs.append(_process.remote(self.tokenizer, data[i:i + step_size], i))

        for i in range(len(ray_objs)):
            processed_docs.extend(ray.get(ray_objs[i]))

        if self.cache_dir:
            print("Saving Into Cache")
            torch.save(processed_docs, cache_path)

        return processed_docs