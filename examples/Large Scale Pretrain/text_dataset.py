import torch
import numpy as np
import logging
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


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

    @staticmethod
    def collate_fn(batch):
        return batch