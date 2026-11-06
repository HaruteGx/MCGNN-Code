from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

BOUNDARIES = (24.0, 26.0)


class JointClassificationRegressionLoss(nn.Module):
    def __init__(self, temperature=0.5, classification_weight=0.5, regression_scale=1.0):
        super().__init__()
        if any(not math.isfinite(x) or x <= 0 for x in (temperature, regression_scale)):
            raise ValueError("Temperature and regression scale must be positive and finite")
        if not math.isfinite(classification_weight) or classification_weight < 0:
            raise ValueError("Classification weight must be nonnegative and finite")
        self.temperature = float(temperature)
        self.classification_weight = float(classification_weight)
        self.regression_scale = float(regression_scale)
        self.register_buffer("boundaries", torch.tensor(BOUNDARIES))

    def per_sample(self, predicted_g, true_g):
        predicted, target = predicted_g.reshape(-1), true_g.reshape(-1)
        if predicted.shape != target.shape or not predicted.numel():
            raise ValueError("Expected matching, nonempty predictions and targets")
        logits = (predicted[:, None] - self.boundaries) / self.temperature
        labels = (target[:, None] >= self.boundaries).to(predicted.dtype)
        classification = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)
        regression = ((predicted - target) / self.regression_scale).square()
        return classification, regression

    def components(self, predicted_g, true_g):
        classification, regression = self.per_sample(predicted_g, true_g)
        return classification.mean(), regression.mean()

    def forward(self, predicted_g, true_g):
        classification, regression = self.components(predicted_g, true_g)
        return regression + self.classification_weight * classification


def classification_weight_at_epoch(target_weight, epoch, warmup_epochs=10):
    if epoch < 1 or warmup_epochs < 0:
        raise ValueError("Epoch must be >=1 and warmup must be >=0")
    return target_weight if warmup_epochs == 0 else target_weight * min(epoch / warmup_epochs, 1.0)


def selection_key(metrics):

    return (-float(metrics["total_loss"]), float(metrics["accuracy"]), float(metrics["macro_f1"]))
