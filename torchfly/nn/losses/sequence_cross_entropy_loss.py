import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
# pylint:disable=no-member


class SequenceCrossEntropyLoss(nn.Module):

    def __init__(self, label_smoothing=-1, ignore_index=-1, reduce=None):
        """
        reduce: None, "mean", "sentence"
        """
        super().__init__()
        self.reduce = reduce
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        if not self.reduce in [None, "none", "mean", "batch", "batch-sequence"]:
            raise NotImplementedError

    def forward(self, logits: torch.FloatTensor, targets: torch.LongTensor, mask=None):
        if mask is None:
            mask = targets != self.ignore_index
        return sequence_cross_entropy_with_logits(logits, targets, mask, self.label_smoothing, self.reduce)


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       mask: torch.BoolTensor,
                                       label_smoothing: bool,
                                       reduce: str = "mean") -> torch.FloatTensor:
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)

    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])

    mask = mask.float()
    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce == "mean":
        loss = loss.sum() / (mask.sum() + 1e-13)
    elif reduce == "batch":
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
    elif reduce == "batch-sequence":
        # we favor longer sequences, so we don't divide with the total sequence length here
        # shape : (batch,)
        loss = loss.sum(1)

    return loss