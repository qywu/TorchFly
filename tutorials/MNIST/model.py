import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

from torchfly.training import FlyModel
from torchfly.metrics import CategoricalAccuracy, Average, MovingAverage, Speed


class CNNNet(nn.Module):
    def __init__(self, conv1_channels, hidden_size):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, 3, 1)
        self.conv2 = nn.Conv2d(conv1_channels, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CNNFlyModel(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = CNNNet(config.model.conv1_channels, config.model.hidden_size)

    def configure_metrics(self):
        self.training_metrics = {"loss": MovingAverage(), "images/s": Speed()}
        self.evaluation_metrics = {"loss": Average(), "acc": CategoricalAccuracy()}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        x, target = batch

        output = self.model(x)
        loss = F.nll_loss(output, target)

        results = {"loss": loss, "output": output}
        self.training_metrics["loss"](loss.item())

        return results

    def predict(self, batch):
        x, target = batch

        output = self.model(x)
        loss = F.nll_loss(output, target)

        self.evaluation_metrics["loss"](loss.item())
        self.evaluation_metrics["acc"](predictions=output, gold_labels=target)
        return None

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
        }
        return metrics

    def get_training_metrics(self) -> Dict[str, str]:
        loss = self.training_metrics["loss"].get_metric()
        metrics = {"loss": f"{loss:.4f}"}
        return metrics

    def get_evaluation_metrics(self) -> Dict[str, str]:
        loss = self.evaluation_metrics["loss"].get_metric()
        acc = self.evaluation_metrics["acc"].get_metric()
        metrics = {"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"}
        return metrics