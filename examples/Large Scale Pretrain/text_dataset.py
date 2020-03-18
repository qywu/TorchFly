import numpy as np
import logging
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# pylint:disable=no-member


class FullDocDataset(IterableDataset):
    def __init__(self, data, max_seq_len: int = 128, sep_token_id: int = -1, allow_cross_doc=True, cycle=False):
        self.data = data
        self.max_seq_len = max_seq_len
        self.sep_token_id = sep_token_id
        self.allow_cross_doc = allow_cross_doc
        self.cycle = cycle

        if isinstance(self.data[0][0], torch.Tensor):
            self.sep_token_id = torch.tensor([self.sep_token_id], dtype=self.data[0][0].dtype)[0]

    def __iter__(self):
        """
        Returns:
            sequence: torch.LongTensor
            document_end: bool whether it is the end of the document
        """
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        seq_buffer = []
        document_end = False
        past_pointer = 0

        # then we sample a sequence with the desired sequence length
        for index in indices:
            document = self.data[index]

            if not self.allow_cross_doc:
                for i in range(0, len(document), self.max_seq_len):
                    document_end = i + self.max_seq_len > len(document)
                    yield self.wrap_output(document[i:i + self.max_seq_len]), document_end
            else:
                while past_pointer < len(document):
                    # history pointer for the current document
                    next_pointer = past_pointer + self.max_seq_len - len(seq_buffer)
                    segment = document[past_pointer:next_pointer]
                    seq_buffer.extend(segment)

                    if len(seq_buffer) == self.max_seq_len:
                        document_end = (past_pointer + self.max_seq_len >= len(document)) | document_end
                        yield self.wrap_output(seq_buffer), document_end
                        seq_buffer = []
                        document_end = False

                    past_pointer += len(segment)

                # if the document is over
                past_pointer = 0
                if len(seq_buffer) > 0:
                    seq_buffer.append(self.sep_token_id)
                    document_end = True

    def wrap_output(self, x):
        x = torch.stack(x, dim=0)
        return x

    def __len__(self):
        "Estimated value. Only use for debug!"
        return sum([len(item) for item in self.data]) // self.max_seq_len


class FullDocCollate:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def __call__(self, batch):
        batch_input_ids, _ = zip(*batch)
        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_input_ids = F.pad(batch_input_ids, (1, 0), value=self.tokenizer.cls_token_id)
        batch_input_ids = F.pad(batch_input_ids, (0, 1), value=self.tokenizer.sep_token_id)
        batch_input_ids = batch_input_ids.long()

        batch_input_ids, labels = self.mask_tokens(batch_input_ids)

        batch = {"input_ids": batch_input_ids, "labels": labels}

        return batch

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.pad_token_id is not None:
            padding_mask = labels.eq(self.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.config.model.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
