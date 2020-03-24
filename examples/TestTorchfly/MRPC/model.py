import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaForSequenceClassification
from torchfly.metrics import Metric, CategoricalAccuracy, F1Measure

from typing import Dict

class InferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._accuracy = CategoricalAccuracy()
        self._f1 = F1Measure(positive_label=1)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch['labels']

        outputs = self.model(input_ids, attention_mask=mask, labels=labels)
        
        results = {
            "loss": outputs[0],
            "outputs": outputs
        }
        return results

    def predict(self, batch):
        results = self.forward(batch)
        self._accuracy(predictions=results["outputs"][1], gold_labels=batch['labels'])
        self._f1(predictions=results["outputs"][1], gold_labels=batch['labels'])
        return None

    def get_metrics(self, reset:bool=False) -> Dict[str, float]:
        precision, recall, f1 = self._f1.get_metric(reset)

        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
            "f1": f1,
        }
        return metrics


def get_model():
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    nn.init.normal_(model.roberta.pooler.dense.weight.data)
    model = InferenceModel(model)
    return model