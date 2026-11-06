import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
    recall_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_DIR / "outputs" / "interpretability/graph_ablations"
DEFAULT_REFERENCE = PROJECT_DIR / "outputs/ir_prediction/tables/ir_models_predictions.csv"
CO_REFERENCE = PROJECT_DIR / "outputs/co_transfer"
CO_FROZEN_REFERENCE = PROJECT_DIR / "config/co_reference"
CO_DIRECTORY = "Co_seed_126_test_14"
TARGET = "G (kcal/mol)"
TRUE_TARGET = f"True {TARGET}"
PREDICTED_TARGET = f"Predicted {TARGET}"
THRESHOLDS = {"Ir": (23.0, 25.0), "Co": (24.0, 26.0)}
BRANCHES = ("environment", "reactant", "product")


def to_classes(values, dataset):
    lower, upper = THRESHOLDS[dataset]
    return np.where(values < lower, 0, np.where(values < upper, 1, 2)).astype(int)


def prediction_metrics(frame, dataset, variant, evaluation):
    if TRUE_TARGET not in frame or PREDICTED_TARGET not in frame:
        raise ValueError(
            f"Prediction file is missing {TRUE_TARGET!r} or {PREDICTED_TARGET!r}."
        )
    truth = pd.to_numeric(frame[TRUE_TARGET], errors="coerce").to_numpy(float)
    prediction = pd.to_numeric(frame[PREDICTED_TARGET], errors="coerce").to_numpy(float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.all():
        raise ValueError(f"Incomplete predictions for {dataset}/{variant}/{evaluation}; no samples may be dropped")
    if not len(truth):
        raise ValueError(f"No finite predictions found for {dataset}/{variant}/{evaluation}.")
    true_classes = to_classes(truth, dataset)
    predicted_classes = to_classes(prediction, dataset)
    matrix = confusion_matrix(true_classes, predicted_classes, labels=[0, 1, 2])
    lower, upper = THRESHOLDS[dataset]
    row = {
        "dataset": dataset,
        "ablation": variant,
        "evaluation": evaluation,
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else float("nan"),
        "accuracy": float(accuracy_score(true_classes, predicted_classes)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_classes, predicted_classes)
        ),
        "macro_f1": float(
            f1_score(true_classes, predicted_classes, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "class_0_recall": float(
            recall_score(true_classes, predicted_classes, labels=[0], average="macro", zero_division=0)
        ),
        "class_1_recall": float(
            recall_score(true_classes, predicted_classes, labels=[1], average="macro", zero_division=0)
        ),
        "class_2_recall": float(
            recall_score(true_classes, predicted_classes, labels=[2], average="macro", zero_division=0)
        ),
        "threshold_1": lower,
        "threshold_2": upper,
    }
    matrix_record = {
        "dataset": dataset,
        "ablation": variant,
        "evaluation": evaluation,
        "labels": [0, 1, 2],
        "confusion_matrix": matrix.tolist(),
        "true_class_counts": np.bincount(true_classes, minlength=3).tolist(),
        "predicted_class_counts": np.bincount(predicted_classes, minlength=3).tolist(),
    }
    return row, matrix_record


def expected_files(root, branch, *, include_co=True):
    ir_root = root / branch
    co_root = root / branch / CO_DIRECTORY / "evaluations/Co_transfer_joint"
    files = (
        ("Ir", "development_oof", ir_root / "five_fold_oof_predictions.csv"),
        ("Ir", "fixed_test", ir_root / "test" / "test_predictions.csv"),
        ("Co", "development_oof", co_root / "development_oof/predictions.csv"),
        ("Co", "holdout_test", co_root / "holdout_test/predictions.csv"),
    )
    return files if include_co else files[:2]


def validate_co_prediction(frame, run, evaluation, variant):

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("development.csv", "test.csv", "development_folds.csv"):
        if digest(run / "split" / name) != digest(CO_FROZEN_REFERENCE / name):
            raise ValueError(f"Co ablation split differs from the main model: {run}/{name}")
    curated_main = variant == "full_model" and Path(run).resolve() == CO_REFERENCE.resolve()
    manifest_path = (run / "split/manifest.json")
    if not manifest_path.exists() and curated_main:
        manifest_path = CO_FROZEN_REFERENCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest["seed"] != 126 or manifest["development_size"] != 125 or manifest["test_size"] != 14:
        raise ValueError("Expected the frozen seed-126 125/14 Co experiment")
    if variant != "full_model":
        inputs = json.loads(((run / "frozen_inputs.json")).read_text(encoding="utf8"))
        if set(inputs["ablate_branches"]) != set(variant.split("_")):
            raise ValueError("Ablation source/checkpoint mask differs from variant")
    receipt_path = (run / "complete.json")
    if receipt_path.exists() or not curated_main:
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        hashes = {k.replace("\\", "/"): v for k,v in receipt["files"].items()}
        relative = f"evaluations/{evaluation}/predictions.csv" if curated_main else f"evaluations/Co_transfer_joint/{evaluation}/predictions.csv"
        if hashes.get(relative) != digest(run / relative):
            raise ValueError("Co prediction is incomplete or differs from the completion receipt")
    source = pd.read_csv((run / "split" / ("development.csv" if evaluation == "development_oof" else "test.csv")))
    if not frame.Name.is_unique or set(frame.Name) != set(source.Name):
        raise ValueError("Co evaluation sample IDs do not match the intended split")
    np.testing.assert_allclose(frame[TRUE_TARGET], source.set_index("Name").loc[frame.Name, TARGET], atol=2e-6, rtol=0)
    if evaluation == "development_oof":
        audit = pd.read_csv((run / "split/development_folds.csv")).set_index("Name")
        np.testing.assert_array_equal(frame.Fold, audit.loc[frame.Name, "fold"])


def summarize(root, branches=BRANCHES, reference_predictions=DEFAULT_REFERENCE,
              *, include_co=True):
    root = Path(root)
    metric_rows = []
    matrix_records = []
    missing = []
    reference_predictions = Path(reference_predictions)
    if reference_predictions.is_file():
        reference = pd.read_csv(reference_predictions)
        reference_specs = (
            ("Ir", "Regress", "development_oof", "development_oof"),
            ("Ir", "Regress", "existing_test", "fixed_test"),
        )
        for dataset, model, source_evaluation, evaluation in reference_specs:
            if dataset == "Co" and not include_co:
                continue
            selected = reference.loc[
                reference["dataset"].eq(dataset)
                & reference["model"].eq(model)
                & reference["evaluation"].eq(source_evaluation)
            ]
            if not selected.empty:
                row, matrix_record = prediction_metrics(
                    selected, dataset, "full_model", evaluation
                )
                metric_rows.append(row)
                matrix_records.append(matrix_record)
    if include_co:
        for evaluation in ("development_oof", "holdout_test"):
            path = CO_REFERENCE / "evaluations/Co_transfer_joint" / evaluation / "predictions.csv"
            if not path.is_file():
                missing.append(str(path))
                continue
            frame = pd.read_csv(path)
            validate_co_prediction(frame, CO_REFERENCE, evaluation, "full_model")
            row, matrix_record = prediction_metrics(frame, "Co", "full_model", evaluation)
            metric_rows.append(row)
            matrix_records.append(matrix_record)
    for branch in branches:
        for dataset, evaluation, path in expected_files(root, branch, include_co=include_co):
            if not path.is_file():
                missing.append(str(path))
                continue
            frame = pd.read_csv(path)
            if dataset == "Co":
                validate_co_prediction(frame, root / branch / CO_DIRECTORY, evaluation, branch)
            row, matrix_record = prediction_metrics(
                frame, dataset, branch, evaluation
            )
            metric_rows.append(row)
            matrix_records.append(matrix_record)
    if not metric_rows:
        raise FileNotFoundError(
            "No ablation prediction files were found. Run run_graph_ablations.py first."
        )
    root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(metric_rows).sort_values(
        ["dataset", "evaluation", "ablation"]
    )
    suffix = "_seed_126" if include_co else ""
    summary.to_csv((root / f"ablation_summary{suffix}.csv"), index=False, encoding="utf-8-sig")
    ((root / f"ablation_confusion_matrices{suffix}.json")).write_text(
        json.dumps(matrix_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if missing:
        ((root / f"incomplete_variants{suffix}.json")).write_text(
            json.dumps({"missing_prediction_files": missing}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif ((root / f"incomplete_variants{suffix}.json")).exists():
        (root / f"incomplete_variants{suffix}.json").unlink()
    return summary, matrix_records, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=BRANCHES)
    parser.add_argument("--reference-predictions", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--skip-co", action="store_true", help="Summarize Ir only.")
    args = parser.parse_args(argv)
    summary, _, missing = summarize(
        args.root, args.branches, args.reference_predictions, include_co=not args.skip_co
    )
    print(summary.to_string(index=False))
    if missing:
        print(f"\nIncomplete variants: {len(missing)} expected prediction files are missing.")
    name = "ablation_summary.csv" if args.skip_co else "ablation_summary_seed_126.csv"
    print(f"\nSaved: {args.root / name}")


if __name__ == "__main__":
    main()
