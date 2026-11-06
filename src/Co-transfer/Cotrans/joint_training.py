from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.train_eval import collate_regression, regression_metrics
from joint_objective import JointClassificationRegressionLoss, BOUNDARIES, selection_key, classification_weight_at_epoch


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def score_predictions(target, predicted, objective):
    target, predicted = np.asarray(target), np.asarray(predicted)
    if not len(target) or not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError("Cannot score empty or nonfinite predictions")
    truth = np.searchsorted(BOUNDARIES, target, side="right")
    classes = np.searchsorted(BOUNDARIES, predicted, side="right")
    cpu_loss = JointClassificationRegressionLoss(objective.temperature, objective.classification_weight, objective.regression_scale)
    with torch.no_grad():
        cls, reg = cpu_loss.components(torch.tensor(predicted, dtype=torch.float64), torch.tensor(target, dtype=torch.float64))
    return {"n": len(target), "correct": int((truth == classes).sum()),
            "accuracy": float(accuracy_score(truth, classes)),
            "macro_f1": float(f1_score(truth, classes, labels=[0, 1, 2], average="macro", zero_division=0)),
            "classification_loss": float(cls), "regression_loss": float(reg),
            "weighted_classification_penalty": float(objective.classification_weight * cls),
            "total_loss": float(reg + objective.classification_weight * cls),
            "mse": float(np.mean((target - predicted) ** 2)), **regression_metrics(target, predicted)}


def train_joint(model, train_data, validation_data, device, *, epochs, learning_rate,
                batch_size=4, weight_decay=1e-4, patience=25, scheduler_patience=10,
                temperature=0.5, classification_weight=0.5, regression_scale=1.0,
                warmup_epochs=10, epoch_offset=0):
    if not train_data or not validation_data or epochs < 1:
        raise ValueError("Expected nonempty data and positive epochs")
    objective = JointClassificationRegressionLoss(temperature, classification_weight, regression_scale).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                         patience=scheduler_patience, min_lr=1e-8)
    loaders = [DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda samples: collate_regression(samples, device))
               for data, shuffle in ((train_data, True), (validation_data, False))]
    best_state, best_metrics, best_epoch, stale, history = None, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        weight = classification_weight_at_epoch(classification_weight, epoch + epoch_offset, warmup_epochs)
        model.train()
        for module in model.children():
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        totals = np.zeros(3)
        for _, environment, reactant, product, temperature_k, targets in loaders[0]:
            prediction = model(environment, reactant, product, temperature_k).reshape(-1)
            cls, reg = objective.components(prediction, targets)
            loss = reg + weight * cls
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite joint loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += np.array([loss.item(), reg.item(), cls.item()]) * len(targets)
        model.eval()
        truth, prediction = [], []
        with torch.no_grad():
            for _, environment, reactant, product, temperature_k, targets in loaders[1]:
                values = model(environment, reactant, product, temperature_k).reshape(-1)
                truth.extend(targets.cpu().tolist())
                prediction.extend(values.cpu().tolist())

        score = score_predictions(truth, prediction, objective)
        scheduler.step(score["total_loss"])
        if best_metrics is None or selection_key(score) > selection_key(best_metrics):
            best_state, best_metrics, best_epoch, stale = cpu_state(model), score, epoch, 0
        else:
            stale += 1
        total, reg, cls = totals / len(train_data)
        history.append({"epoch": epoch, "classification_weight": weight, "train_loss": total,
                        "train_regression_loss": reg, "train_classification_loss": cls,
                        "train_weighted_classification_penalty": weight * cls,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        **{f"validation_{key}": value for key, value in score.items()}})
        print(f"Epoch {epoch}/{epochs} | reg={reg:.5f} cls={cls:.5f} weight={weight:.3f} "
              f"val joint={score['total_loss']:.5f} ACC={score['accuracy']:.4f} MAE={score['mae']:.4f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state, strict=True)
    return {"history": history, "best_epoch": best_epoch, "best_metrics": best_metrics}
