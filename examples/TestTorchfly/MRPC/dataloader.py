import os
import torch
import logging

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class CycleDataloader(torch.utils.data.DataLoader):
    """
    A warpper to DataLoader that the `__next__` method will never end. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = None

    def __iter__(self):
        for batch in super().__iter__():
            batch = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            yield batch
    def __next__(self):
        if self.iterator is None:
            self.iterator = self.__iter__()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = self.__iter__()
            batch = next(self.iterator)
        return batch


def load_and_cache_examples(config, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        config.task.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, config.task.model_name.split("/"))).pop(),
            str(config.task.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not config.task.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", config.task.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and config.task.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(config.task.data_dir)
            if evaluate else processor.get_train_examples(config.task.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=config.task.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(config.task.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if config.task.model_type in ["xlnet"] else 0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def get_data_loader(config, evaluate=False):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_dataset = load_and_cache_examples(config, config.task.task_name, tokenizer, evaluate=evaluate)
    train_sampler = RandomSampler(train_dataset) if config.training.num_gpus_per_node == 1 or evaluate else DistributedSampler(train_dataset)
    train_dataloader = CycleDataloader(train_dataset, sampler=train_sampler, batch_size=config.training.batch_size)

    return train_dataloader