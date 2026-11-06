import json
from pathlib import Path

import numpy as np

from .data_splits import checkpoint_split_config, record_ids, sha256_file, LEGACY_SPLIT_VERSION
from .prediction import load_regression_checkpoint, predict_dataframe
from .train_eval import regression_metrics, save_regression_scatter
from .regression_classification import export_classification


def evaluate_frozen_holdout(checkpoint_path, test_frame, split_manifest, device,
                            output_dir, batch_size=4, completed_run_dir=None,
                            development_path=None):

    if record_ids(test_frame).tolist() != split_manifest["test_names"]:
        raise ValueError("Evaluation rows differ from the frozen test set.")
    model, metadata, _ = load_regression_checkpoint(checkpoint_path, device)
    existing_run_evidence = None
    if metadata.get("holdout_split") != checkpoint_split_config(split_manifest):
        if (split_manifest.get("split_version") == LEGACY_SPLIT_VERSION
                and metadata.get("holdout_split") is None
                and completed_run_dir is not None and development_path is not None):
            from .completed_run_audit import verify_completed_class_run
            existing_run_evidence = verify_completed_class_run(
                completed_run_dir, checkpoint_path, development_path, split_manifest)
        else:
            raise ValueError(
                "Checkpoint was not trained under this frozen Ir holdout protocol. "
                "Retrain using the current seed and test fraction; checkpoints from "
                "Class-only, different-seed or all-Ir training cannot evaluate this random holdout."
            )
    target_column = metadata.get("target_column", "G (kcal/mol)")
    targets = test_frame[target_column].to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Frozen test targets must all be finite.")
    checkpoint_hash = sha256_file(checkpoint_path)
    results, _ = predict_dataframe(model, test_frame, device,
                                   batch_size=batch_size, default_catalyst="Ir",
                                   target_column=target_column)
    prediction_column = f"Predicted {target_column}"
    if not results["status"].eq("OK").all() or not np.isfinite(results[prediction_column]).all():
        failures = results.loc[results["status"].ne("OK"), ["Name", "reason"]]
        raise RuntimeError(f"Some frozen test rows could not be predicted: {failures.to_dict('records')}")
    predictions = results[prediction_column].to_numpy(dtype=float)
    metrics = regression_metrics(targets, predictions)
    results[f"True {target_column}"] = targets
    results["Residual (Predicted - True)"] = predictions-targets
    results["Absolute Error"] = np.abs(predictions-targets)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = export_classification(results, output, "Ir", stem="test_classification",
                                    role="Ir fixed holdout test: regression to classes")
    results.to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "purpose": metadata.get("purpose", "fixed holdout evaluation"),
        "dataset_role": "fixed_holdout_test",
        "test_rows": len(test_frame), "target_column": target_column,
        "metrics": metrics, "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_hash, "selected_cv_fold": metadata.get("cv_fold"),
        "holdout_split": checkpoint_split_config(split_manifest),
        "split_overlap_audit": split_manifest.get("audit", {}),
        "existing_run_provenance": existing_run_evidence,
        "xtb_test_coverage_rows": {column: int(results[column].sum())
                                   for column in results.columns
                                   if column.startswith("xtb_") and column.endswith("_available")},
        "note": "Test labels were not used for training, early stopping, learning-rate scheduling or checkpoint selection.",
    }
    (output / "test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_regression_scatter(targets, predictions, output / "test_scatter.png",
                            "Frozen Ir holdout test", target_label=target_column)
    print(f"Independent Ir test ({len(test_frame)} rows): {metrics}", flush=True)
    return report, results
