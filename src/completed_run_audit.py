from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_splits import record_ids, sha256_file, LEGACY_SPLIT_VERSION


def verify_completed_class_run(run_dir, checkpoint_path, development_path, manifest):

    if manifest.get("split_version") != LEGACY_SPLIT_VERSION:
        raise ValueError("Legacy Class provenance cannot be reused for a random Ir holdout; retrain.")
    run = Path(run_dir).resolve()
    selected = Path(checkpoint_path).resolve()
    summary_path = run / "five_fold_cross_validation_summary.xlsx"
    if not summary_path.is_file():
        raise FileNotFoundError("Five-fold training is not complete: summary workbook is missing.")
    if sha256_file(development_path) != manifest["sources"]["development"]["sha256"]:
        raise ValueError("Class source differs from the frozen holdout protocol.")
    development = pd.read_csv(development_path)
    names = record_ids(development).to_numpy()
    summary = pd.read_excel(summary_path, sheet_name="Fold Metrics")
    folds = summary["fold"].astype(int).tolist()
    if sorted(folds) != [1, 2, 3, 4, 5]:
        raise ValueError("Exactly five completed folds are required before test evaluation.")
    winner = int(summary.sort_values(
        ["validation_r2", "validation_rmse"], ascending=[False, True], kind="stable").iloc[0]["fold"])
    winner_path = run / f"fold_{winner}" / "ir_xtb_metal_regression_checkpoint.pth"
    if not selected.is_file() or not winner_path.is_file() or sha256_file(selected) != sha256_file(winner_path):
        raise ValueError("Selected model is not the saved validation-only best fold.")
    evidence = []
    validation_positions = []
    for fold in range(1, 6):
        folder = run / f"fold_{fold}"
        path = folder / "ir_xtb_metal_regression_checkpoint.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Fold {fold} has no completed checkpoint.")
        if path.stat().st_mtime_ns > summary_path.stat().st_mtime_ns:
            raise ValueError("Summary is older than a fold checkpoint; this run may still be in progress.")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        if (metadata.get("task") != "reaction_energy_regression"
                or metadata.get("target_column") != "G (kcal/mol)"
                or metadata.get("n_splits") != 5 or metadata.get("cv_fold") != fold
                or Path(metadata.get("data_path", "")).resolve() != Path(development_path).resolve()):
            raise ValueError(f"Fold {fold} does not identify the current Class regression data.")
        source = np.asarray(metadata.get("valid_source_rows", []), dtype=int)
        train = np.asarray(metadata.get("train_indices", []), dtype=int)
        valid = np.asarray(metadata.get("validation_indices", []), dtype=int)
        if (sorted(source.tolist()) != list(range(len(development)))
                or sorted(np.concatenate([train, valid]).tolist()) != list(range(len(source)))):
            raise ValueError(f"Fold {fold} is not a complete disjoint Class training/validation partition.")
        for role, positions in (("train", train), ("validation", valid)):
            result_path = folder / f"{role}_regression_results.xlsx"
            result = pd.read_excel(result_path, sheet_name="Predictions", dtype={"Sample Name": str})
            expected_names = names[source[positions]].tolist()
            if result["Sample Name"].astype(str).tolist() != expected_names:
                raise ValueError(f"Fold {fold} {role} identities do not match the frozen Class rows.")
            expected_targets = development.iloc[source[positions]]["G (kcal/mol)"].to_numpy(dtype=float)
            if not np.allclose(result["True G (kcal/mol)"].to_numpy(dtype=float),
                               expected_targets, rtol=0, atol=5e-6):
                raise ValueError(f"Fold {fold} {role} labels differ from the frozen Class labels.")
            if set(expected_names) & set(manifest["test_names"]):
                raise ValueError(f"Fold {fold} includes frozen test records.")
        validation_positions.extend(source[valid].tolist())
        evidence.append({"fold": fold, "train_rows": len(train), "validation_rows": len(valid),
                         "checkpoint_sha256": sha256_file(path)})
        del payload
    if sorted(validation_positions) != list(range(len(development))):
        raise ValueError("Completed folds do not provide exactly one validation prediction per Class row.")
    return {
        "verification": "all five fold Names, source indices and true targets verified against frozen Class data",
        "selected_cv_fold": winner, "selected_checkpoint_sha256": sha256_file(selected),
        "development_sha256": manifest["sources"]["development"]["sha256"],
        "test_record_overlap": 0, "folds": evidence,
    }
