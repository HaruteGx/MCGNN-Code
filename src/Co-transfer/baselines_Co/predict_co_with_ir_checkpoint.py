import argparse
import hashlib
import json
import os
import sys

import pandas as pd
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "Co-transfer"))
SRC_DIR = PROJECT_DIR
normalized_src = os.path.normcase(os.path.abspath(SRC_DIR))
sys.path[:] = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or os.curdir)) != normalized_src
]
sys.path.insert(0, SRC_DIR)
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

from src.ir_protocol import ir_run_paths
from src.prediction import load_regression_checkpoint, predict_dataframe
from src.train_eval import regression_metrics, save_regression_scatter
from src.xtb_features import get_xtb_store


PROTOCOL_PATHS = ir_run_paths(PROJECT_DIR)
DEFAULT_CHECKPOINT = str(PROTOCOL_PATHS["stable_checkpoint"])
DEFAULT_DATA = os.path.join(PROJECT_DIR, "data", "Database Co.csv")
DEFAULT_XTB_ROOT = os.path.join(PROJECT_DIR, "data", "xtb_Co_features")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "baselines_Co_old",
    "full_data_zero_shot",
    "ir_zero_shot_to_co",
)
TARGET_COLUMN = "G (kcal/mol)"
PHYSICS_HEAD = "enthalpy_entropy_physics"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path):
    extension = os.path.splitext(path)[1].lower()
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if extension == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported Co data file: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use the frozen Ir Delta-H/Delta-S model to predict Co directly."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--xtb_root", default=DEFAULT_XTB_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow_missing_indices", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            "Frozen Ir checkpoint not found. Train the current Ir model first: "
            f"{args.checkpoint}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model, metadata, model_config = load_regression_checkpoint(
        args.checkpoint, device
    )
    if metadata.get("head") != PHYSICS_HEAD:
        raise ValueError(
            "The Ir checkpoint is not the current two-output Delta-H/Delta-S model."
        )
    if metadata.get("default_catalyst") not in (None, "Ir"):
        raise ValueError("The selected source checkpoint is not an Ir model.")

    data = load_table(args.data)
    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"Co baseline data is missing {TARGET_COLUMN!r}.")
    result, _ = predict_dataframe(
        model,
        data,
        device,
        batch_size=args.batch_size,
        allow_missing_indices=args.allow_missing_indices,
        default_catalyst="Co",
        target_column=TARGET_COLUMN,
        xtb_root=args.xtb_root,
    )
    predicted_column = f"Predicted {TARGET_COLUMN}"
    result["Residual (Predicted - True)"] = (
        result[predicted_column] - result[TARGET_COLUMN]
    )
    result["Absolute Error"] = result["Residual (Predicted - True)"].abs()
    valid = result[
        result["status"].eq("OK")
        & result[TARGET_COLUMN].notna()
        & result[predicted_column].notna()
    ].copy()
    if len(valid) < 2:
        raise RuntimeError("Fewer than two Co rows received valid zero-shot predictions.")
    predictions = valid[predicted_column].to_numpy(dtype=float)
    targets = valid[TARGET_COLUMN].to_numpy(dtype=float)
    metrics = regression_metrics(targets, predictions)

    csv_path = os.path.join(args.output_dir, "ir_zero_shot_co_predictions.csv")
    from src.regression_classification import export_classification
    result = export_classification(result, args.output_dir, "Co", stem="zero_shot_classification",
                                   role="Frozen Ir GNN: zero-shot Co predictions")
    xlsx_path = os.path.join(args.output_dir, "ir_zero_shot_co_predictions.xlsx")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="All Predictions", index=False)
        valid.to_excel(writer, sheet_name="Evaluated Rows", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics", index=False)
    scatter_path = os.path.join(args.output_dir, "ir_zero_shot_co_scatter.png")
    save_regression_scatter(
        targets,
        predictions,
        scatter_path,
        "Frozen Ir checkpoint -> Co (zero-shot)",
        target_label=TARGET_COLUMN,
    )

    xtb_report = (
        get_xtb_store(args.xtb_root).report()
        if model.xtb_feature_config.get("enabled", True)
        else None
    )
    report = {
        "task": "frozen_ir_checkpoint_zero_shot_prediction_on_co",
        "co_weights_fitted": False,
        "co_labels_used_for_training_or_selection": False,
        "co_labels_used_for_final_metrics_only": True,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_task": metadata.get("task"),
        "source_cv_fold": metadata.get("cv_fold"),
        "head": metadata.get("head"),
        "thermodynamic_relation": metadata.get("thermodynamic_relation"),
        "model_config": model_config,
        "co_data": os.path.abspath(args.data),
        "co_data_sha256": sha256_file(args.data),
        "co_rows": int(len(data)),
        "predicted_rows": int(len(valid)),
        "failed_rows": int(len(data) - len(valid)),
        "xtb_data_provenance": xtb_report,
        "metrics": metrics,
    }
    with open(
        os.path.join(args.output_dir, "ir_zero_shot_co_report.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    print("Ir -> Co zero-shot baseline complete.")
    print(f"Predicted rows: {len(valid)}/{len(data)}")
    print(f"Metrics: {metrics}")
    print(f"Predictions: {csv_path}")


if __name__ == "__main__":
    main()
