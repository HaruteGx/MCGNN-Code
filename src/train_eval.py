import os
from typing import Dict, Optional, Sequence, Tuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader


def collate_regression(samples, device):
    (
        names,
        environment_graphs,
        reactant_graphs,
        product_graphs,
        temperatures_kelvin,
        targets,
    ) = zip(*samples)
    return (
        names,
        dgl.batch(environment_graphs).to(device),
        dgl.batch(reactant_graphs).to(device),
        dgl.batch(product_graphs).to(device),
        torch.tensor(temperatures_kelvin, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def regression_metrics(targets: Sequence[float], predictions: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(targets, dtype=float).reshape(-1)
    y_pred = np.asarray(predictions, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: targets {y_true.shape} vs predictions {y_pred.shape}")
    residuals = y_pred - y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def _copy_state_to_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_regressor(
    model,
    train_data,
    validation_data,
    device,
    num_epochs=100,
    batch_size=2,
    learning_rate=1e-4,
    weight_decay=1e-4,
    scheduler_patience=7,
    scheduler_factor=0.40,
    minimum_learning_rate=1e-8,
    early_stopping_patience: Optional[int] = None,
    min_delta=1e-4,
    restore_best_weights=True,
):


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=minimum_learning_rate,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda samples: collate_regression(samples, device),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate_regression(samples, device),
    )

    history = {
        "train_loss": [],
        "validation_r2": [],
        "validation_rmse": [],
        "validation_mae": [],
        "learning_rate": [],
    }
    best_validation_r2 = -float("inf")
    best_validation_rmse = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, int(num_epochs) + 1):
        model.train()

        for module in model.children():
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
        total_squared_error = 0.0
        for _, environment, reactant, product, temperature, targets in train_loader:


            predictions = model(
                environment, reactant, product, temperature
            ).squeeze(-1)
            loss = criterion(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_squared_error += float(loss.item()) * targets.size(0)

        average_loss = total_squared_error / len(train_data)
        model.eval()
        validation_targets = []
        validation_predictions = []
        with torch.no_grad():
            for _, environment, reactant, product, temperature, targets in validation_loader:
                predictions = model(
                    environment, reactant, product, temperature
                ).squeeze(-1)
                validation_targets.extend(targets.detach().cpu().numpy().reshape(-1))
                validation_predictions.extend(
                    predictions.detach().cpu().numpy().reshape(-1)
                )

        metrics = regression_metrics(validation_targets, validation_predictions)
        scheduler.step(metrics["r2"])
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(average_loss))
        history["validation_r2"].append(metrics["r2"])
        history["validation_rmse"].append(metrics["rmse"])
        history["validation_mae"].append(metrics["mae"])
        history["learning_rate"].append(current_learning_rate)

        improved = metrics["r2"] > best_validation_r2 + min_delta
        tied_but_lower_rmse = (
            abs(metrics["r2"] - best_validation_r2) <= min_delta
            and metrics["rmse"] < best_validation_rmse
        )
        if improved or tied_but_lower_rmse:
            best_validation_r2 = metrics["r2"]
            best_validation_rmse = metrics["rmse"]
            best_epoch = epoch
            best_state = _copy_state_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch [{epoch}/{num_epochs}] | Train MSE: {average_loss:.4f} | "
            f"Validation R2: {metrics['r2']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | "
            f"LR: {current_learning_rate:.6g}"
        )
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= int(early_stopping_patience)
        ):
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation R2: {best_validation_r2:.4f} at epoch {best_epoch}."
            )
            break

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_validation_r2"] = best_validation_r2
    history["best_validation_rmse"] = best_validation_rmse
    return history


def predict_regressor(
    model, data_split, device, batch_size=2, return_components=False
) -> Tuple[list, np.ndarray, np.ndarray]:
    loader = DataLoader(
        data_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate_regression(samples, device),
    )
    model.eval()
    names_all = []
    targets_all = []
    predictions_all = []
    temperatures_all = []
    enthalpies_all = []
    entropies_all = []
    with torch.no_grad():
        for names, environment, reactant, product, temperature, targets in loader:
            predictions, enthalpies, entropies = model(
                environment,
                reactant,
                product,
                temperature,
                return_components=True,
            )
            names_all.extend(names)
            targets_all.extend(targets.detach().cpu().numpy().reshape(-1))
            predictions_all.extend(
                predictions.detach().cpu().numpy().reshape(-1)
            )
            temperatures_all.extend(
                temperature.detach().cpu().numpy().reshape(-1)
            )
            enthalpies_all.extend(
                enthalpies.detach().cpu().numpy().reshape(-1)
            )
            entropies_all.extend(
                entropies.detach().cpu().numpy().reshape(-1)
            )
    result = (
        names_all,
        np.asarray(targets_all, dtype=float),
        np.asarray(predictions_all, dtype=float),
    )
    if not return_components:
        return result
    components = {
        "temperature_kelvin": np.asarray(temperatures_all, dtype=float),
        "delta_h_kcal_mol": np.asarray(enthalpies_all, dtype=float),
        "delta_s_cal_mol_k": np.asarray(entropies_all, dtype=float),
    }
    return (*result, components)


def save_regression_scatter(
    targets,
    predictions,
    save_path,
    title,
    target_label="Target",
):
    metrics = regression_metrics(targets, predictions)
    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    minimum = float(min(y_true.min(), y_pred.min()))
    maximum = float(max(y_true.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(y_true, y_pred, alpha=0.70, s=18)
    ax.plot([minimum, maximum], [minimum, maximum], "k--", linewidth=1.2)
    ax.set_xlabel(f"True {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(
        f"{title}\nR2={metrics['r2']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
    )
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_save(
    model,
    data_split,
    device,
    save_xlsx_path,
    title_prefix,
    batch_size=2,
    target_column="G (kcal/mol)",
    classification_dataset="Ir",
):
    names, targets, predictions, components = predict_regressor(
        model,
        data_split,
        device,
        batch_size=batch_size,
        return_components=True,
    )
    metrics = regression_metrics(targets, predictions)
    result = pd.DataFrame(
        {
            "Sample Name": names,
            "Temperature (K)": components["temperature_kelvin"],
            "Predicted Delta H (kcal/mol)": components["delta_h_kcal_mol"],
            "Predicted Delta S (cal/mol/K)": components["delta_s_cal_mol_k"],
            f"True {target_column}": targets,
            f"Predicted {target_column}": predictions,
            "Residual (Predicted - True)": predictions - targets,
            "Absolute Error": np.abs(predictions - targets),
        }
    )
    os.makedirs(os.path.dirname(save_xlsx_path) or ".", exist_ok=True)
    from .regression_classification import export_classification
    result = export_classification(
        result, os.path.dirname(save_xlsx_path) or ".", classification_dataset,
        stem=os.path.splitext(os.path.basename(save_xlsx_path))[0] + "_classification",
        role=title_prefix,
    )
    with pd.ExcelWriter(save_xlsx_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics", index=False)
    plot_path = os.path.splitext(save_xlsx_path)[0] + "_scatter.png"
    save_regression_scatter(
        targets,
        predictions,
        plot_path,
        title_prefix,
        target_label=target_column,
    )
    print(
        f"{title_prefix} R2: {metrics['r2']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}"
    )
    print(f"{title_prefix} regression results saved to {save_xlsx_path}")
    return metrics, result


def plot_training_history(history, save_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axes[0].plot(epochs, history["train_loss"], color="tab:blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training MSE")
    axes[0].set_title("Training Loss")
    axes[1].plot(epochs, history["validation_r2"], color="tab:orange")
    if history.get("best_epoch"):
        best_epoch = int(history["best_epoch"])
        axes[1].scatter(
            [best_epoch],
            [history["validation_r2"][best_epoch - 1]],
            color="tab:red",
            label=f"Best epoch {best_epoch}",
            zorder=3,
        )
        axes[1].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation R2")
    axes[1].set_title("Validation R2")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
