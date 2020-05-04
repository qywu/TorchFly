from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

from torchfly.training import FlyModule
from torchfly.metrics import CategoricalAccuracy


class CNNNet(FlyModule):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self._accuracy = CategoricalAccuracy()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        x = batch["input"]
        target = batch["target"]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        loss = F.nll_loss(output, target)

        results = {
            "loss": loss,
            "output": output
        }

        return results

    def predict(self, batch):
        results = self.forward(batch)
        self._accuracy(predictions=results["output"], gold_labels=batch['target'])
        return None

    def get_metrics(self, reset:bool=False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
        }
        return metrics