import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaForSequenceClassification
from torchfly.metrics import Metric, CategoricalAccuracy, F1Measure

from typing import Dict

from exbert_model import ExBertForMaskedLM


class PretrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ExBertForMaskedLM(config.model)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        masked_lm_labels = batch["labels"]
        results = self.model(input_ids=input_ids, masked_lm_labels=masked_lm_labels)
        return results

    def predict(self, batch):
        "For inference"
        pass

    def get_metrics(self, reset):
        return {}