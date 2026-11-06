from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baselines.ML.train_rdkit_baselines import (
    DESCRIPTOR_CONFORMER_SEED,
    DESCRIPTOR_NUM_CONFORMERS,
    ENSEMBLE_TREES,
    ENVIRONMENT_FINGERPRINT_BITS,
    FEATURE_VERSION,
    FORBIDDEN_MODEL_COLUMNS,
    NAME_COLUMN,
    OPTIMIZE_CONFORMERS,
    QUICK_ENSEMBLE_TREES,
    REACTION_FINGERPRINT_BITS,
    ROLE_COLUMNS,
    TARGET_COLUMN,
    SELECTED_FEATURES,
    _tree_model,
    _read_table,
    _sha256,
    bounded_native_threads,
    feature_names,
    load_or_build_features,
    make_co_splits,
    publication_sources,
    regression_metrics,
)


METHOD_ORDER = (
    "IrOnly_ZeroShot_RF",
    "CoOnly_RF",
    "WeightedPooled_RF",
    "IrResidualCo_RF",
)
DEFAULT_SOURCE_WEIGHT_GRID = (0.0, 0.05, 0.20)
DEFAULT_RESIDUAL_ETA_GRID = (0.0, 0.5, 1.0)
DEFAULT_SELECTION_TREES = 40
QUICK_SELECTION_TREES = 8


def build_random_forest(
    *,
    seed: int,
    n_estimators: int = ENSEMBLE_TREES,
    n_jobs: int = 1,
    selected_features: int = SELECTED_FEATURES,
) -> Pipeline:

    return _tree_model(
        RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=10,
            max_leaf_nodes=64,
            max_features=0.35,
            min_samples_leaf=2,
            n_jobs=int(n_jobs),
            random_state=int(seed),
        ),
        selected_features=int(selected_features),
    )


def parse_nonnegative_grid(text: str, *, argument_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{argument_name} must be a comma-separated numeric grid.") from error
    if not values:
        raise ValueError(f"{argument_name} cannot be empty.")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{argument_name} values must be finite and non-negative.")
    if len(set(values)) != len(values):
        raise ValueError(f"{argument_name} contains duplicate values.")
    return tuple(sorted(values))


def inner_splits_from_outer_assignment(
    outer_train_indices: np.ndarray,
    fold_assignments: np.ndarray,
    reaction_groups: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:

    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    fold_assignments = np.asarray(fold_assignments, dtype=int)
    reaction_groups = np.asarray(reaction_groups, dtype=object)
    if len(fold_assignments) != len(reaction_groups):
        raise ValueError("Fold assignments and reaction groups have different lengths.")
    if len(outer_train_indices) == 0:
        raise ValueError("Outer training indices cannot be empty.")

    inner_splits = []
    for validation_fold in sorted(set(fold_assignments[outer_train_indices].tolist())):
        validation_mask = fold_assignments[outer_train_indices] == validation_fold
        inner_validation = outer_train_indices[validation_mask]
        inner_train = outer_train_indices[~validation_mask]
        if len(inner_train) == 0 or len(inner_validation) == 0:
            raise ValueError("Every inner split must contain training and validation rows.")
        overlap = set(reaction_groups[inner_train]) & set(reaction_groups[inner_validation])
        if overlap:
            raise RuntimeError(
                f"Reaction groups cross an inner split: {sorted(overlap)[:5]}"
            )
        inner_splits.append((int(validation_fold), inner_train, inner_validation))
    if len(inner_splits) < 2:
        raise ValueError("At least two inner validation blocks are required.")
    return inner_splits


def _fit_forest(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    native_threads: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Pipeline:
    model = build_random_forest(
        seed=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    with warnings.catch_warnings(), bounded_native_threads(native_threads):
        warnings.simplefilter("ignore")
        fit_parameters = (
            {"regressor__sample_weight": sample_weight}
            if sample_weight is not None
            else {}
        )
        model.fit(features, targets, **fit_parameters)
    return model


def _fit_weighted_pooled_forest(
    ir_features: np.ndarray,
    ir_targets: np.ndarray,
    co_features: np.ndarray,
    co_targets: np.ndarray,
    *,
    source_weight: float,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    native_threads: int,
) -> Pipeline:
    if source_weight == 0.0:
        return _fit_forest(
            co_features,
            co_targets,
            seed=seed,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            native_threads=native_threads,
        )
    pooled_features = np.concatenate((ir_features, co_features), axis=0)
    pooled_targets = np.concatenate((ir_targets, co_targets), axis=0)
    sample_weights = np.concatenate(
        (
            np.full(len(ir_targets), float(source_weight), dtype=float),
            np.ones(len(co_targets), dtype=float),
        )
    )
    return _fit_forest(
        pooled_features,
        pooled_targets,
        seed=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        native_threads=native_threads,
        sample_weight=sample_weights,
    )


def select_source_weight(
    ir_features: np.ndarray,
    ir_targets: np.ndarray,
    co_features: np.ndarray,
    co_targets: np.ndarray,
    inner_splits: list[tuple[int, np.ndarray, np.ndarray]],
    candidates: tuple[float, ...],
    *,
    outer_fold: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    native_threads: int,
) -> tuple[float, list[dict[str, object]]]:

    rows = []
    for source_weight in candidates:
        validation_targets = []
        validation_predictions = []
        for inner_fold, inner_train, inner_validation in inner_splits:
            model = _fit_weighted_pooled_forest(
                ir_features,
                ir_targets,
                co_features[inner_train],
                co_targets[inner_train],
                source_weight=source_weight,
                seed=seed + 10_000 * outer_fold + 100 * inner_fold,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                native_threads=native_threads,
            )
            validation_targets.append(co_targets[inner_validation])
            validation_predictions.append(model.predict(co_features[inner_validation]))
        pooled_targets = np.concatenate(validation_targets)
        pooled_predictions = np.concatenate(validation_predictions)
        metrics = regression_metrics(pooled_targets, pooled_predictions)
        rows.append(
            {
                "method": "WeightedPooled_RF",
                "outer_fold": int(outer_fold),
                "parameter": "ir_sample_weight",
                "candidate_value": float(source_weight),
                "inner_validation_rows": int(len(pooled_targets)),
                "inner_validation_blocks": int(len(inner_splits)),
                "inner_r2": metrics["r2"],
                "inner_rmse": metrics["rmse"],
                "inner_mae": metrics["mae"],
                "selected": False,
            }
        )
    selected_row = min(rows, key=lambda row: (row["inner_rmse"], row["candidate_value"]))
    selected_row["selected"] = True
    return float(selected_row["candidate_value"]), rows


def select_residual_eta(
    source_predictions: np.ndarray,
    co_features: np.ndarray,
    co_targets: np.ndarray,
    inner_splits: list[tuple[int, np.ndarray, np.ndarray]],
    candidates: tuple[float, ...],
    *,
    outer_fold: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    native_threads: int,
) -> tuple[float, list[dict[str, object]]]:

    covered_indices = []
    residual_predictions = []
    for inner_fold, inner_train, inner_validation in inner_splits:
        residual_targets = co_targets[inner_train] - source_predictions[inner_train]
        model = _fit_forest(
            co_features[inner_train],
            residual_targets,
            seed=seed + 20_000 * outer_fold + 100 * inner_fold,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            native_threads=native_threads,
        )
        covered_indices.append(inner_validation)
        residual_predictions.append(model.predict(co_features[inner_validation]))

    covered = np.concatenate(covered_indices)
    corrections = np.concatenate(residual_predictions)
    rows = []
    for eta in candidates:
        predictions = source_predictions[covered] + float(eta) * corrections
        metrics = regression_metrics(co_targets[covered], predictions)
        rows.append(
            {
                "method": "IrResidualCo_RF",
                "outer_fold": int(outer_fold),
                "parameter": "residual_correction_eta",
                "candidate_value": float(eta),
                "inner_validation_rows": int(len(covered)),
                "inner_validation_blocks": int(len(inner_splits)),
                "inner_r2": metrics["r2"],
                "inner_rmse": metrics["rmse"],
                "inner_mae": metrics["mae"],
                "selected": False,
            }
        )
    selected_row = min(rows, key=lambda row: (row["inner_rmse"], row["candidate_value"]))
    selected_row["selected"] = True
    return float(selected_row["candidate_value"]), rows


def _fold_metric_row(
    method: str,
    fold: int,
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    ir_train_rows: int,
    co_train_rows: int,
    source_weight: Optional[float] = None,
    residual_eta: Optional[float] = None,
) -> dict[str, object]:
    return {
        "dataset": "Co",
        "evaluation": "reaction_grouped_5fold_oof",
        "method": method,
        "fold": int(fold),
        "ir_train_rows": int(ir_train_rows),
        "co_train_rows": int(co_train_rows),
        "validation_rows": int(len(targets)),
        "selected_ir_sample_weight": source_weight,
        "selected_residual_eta": residual_eta,
        **regression_metrics(targets, predictions),
    }


def evaluate_transfer_methods(
    ir_features: np.ndarray,
    ir_targets: np.ndarray,
    co_frame: pd.DataFrame,
    co_features: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    folds: np.ndarray,
    groups: np.ndarray,
    *,
    source_weight_grid: tuple[float, ...],
    residual_eta_grid: tuple[float, ...],
    seed: int,
    n_estimators: int,
    selection_estimators: int,
    n_jobs: int,
    native_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    co_targets = co_frame[TARGET_COLUMN].to_numpy(dtype=float)
    method_predictions = {
        method: np.full(len(co_frame), np.nan, dtype=float) for method in METHOD_ORDER
    }
    selected_source_weight = np.full(len(co_frame), np.nan, dtype=float)
    selected_residual_eta = np.full(len(co_frame), np.nan, dtype=float)
    correction_component = np.full(len(co_frame), np.nan, dtype=float)
    method_seconds = {method: 0.0 for method in METHOD_ORDER}
    fold_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    print("[RF transfer] fitting Ir source forest", flush=True)
    started = time.perf_counter()
    source_model = _fit_forest(
        ir_features,
        ir_targets,
        seed=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        native_threads=native_threads,
    )
    source_predictions = np.asarray(source_model.predict(co_features), dtype=float)
    method_seconds["IrOnly_ZeroShot_RF"] = time.perf_counter() - started
    method_predictions["IrOnly_ZeroShot_RF"][:] = source_predictions

    for fold, (outer_train, outer_validation) in enumerate(splits, start=1):
        print(f"[RF transfer] Co outer fold {fold}/5", flush=True)
        inner_splits = inner_splits_from_outer_assignment(outer_train, folds, groups)

        started = time.perf_counter()
        co_model = _fit_forest(
            co_features[outer_train],
            co_targets[outer_train],
            seed=seed,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            native_threads=native_threads,
        )
        co_prediction = np.asarray(co_model.predict(co_features[outer_validation]), dtype=float)
        method_seconds["CoOnly_RF"] += time.perf_counter() - started
        method_predictions["CoOnly_RF"][outer_validation] = co_prediction

        started = time.perf_counter()
        source_weight, rows = select_source_weight(
            ir_features,
            ir_targets,
            co_features,
            co_targets,
            inner_splits,
            source_weight_grid,
            outer_fold=fold,
            seed=seed,
            n_estimators=selection_estimators,
            n_jobs=n_jobs,
            native_threads=native_threads,
        )
        selection_rows.extend(rows)
        pooled_model = _fit_weighted_pooled_forest(
            ir_features,
            ir_targets,
            co_features[outer_train],
            co_targets[outer_train],
            source_weight=source_weight,
            seed=seed,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            native_threads=native_threads,
        )
        pooled_prediction = np.asarray(
            pooled_model.predict(co_features[outer_validation]), dtype=float
        )
        method_seconds["WeightedPooled_RF"] += time.perf_counter() - started
        method_predictions["WeightedPooled_RF"][outer_validation] = pooled_prediction
        selected_source_weight[outer_validation] = source_weight

        started = time.perf_counter()
        residual_eta, rows = select_residual_eta(
            source_predictions,
            co_features,
            co_targets,
            inner_splits,
            residual_eta_grid,
            outer_fold=fold,
            seed=seed,
            n_estimators=selection_estimators,
            n_jobs=n_jobs,
            native_threads=native_threads,
        )
        selection_rows.extend(rows)
        residual_targets = co_targets[outer_train] - source_predictions[outer_train]
        residual_model = _fit_forest(
            co_features[outer_train],
            residual_targets,
            seed=seed,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            native_threads=native_threads,
        )
        fold_correction = np.asarray(
            residual_model.predict(co_features[outer_validation]), dtype=float
        )
        residual_prediction = (
            source_predictions[outer_validation] + residual_eta * fold_correction
        )
        method_seconds["IrResidualCo_RF"] += time.perf_counter() - started
        method_predictions["IrResidualCo_RF"][outer_validation] = residual_prediction
        selected_residual_eta[outer_validation] = residual_eta
        correction_component[outer_validation] = fold_correction

        per_fold_predictions = {
            "IrOnly_ZeroShot_RF": source_predictions[outer_validation],
            "CoOnly_RF": co_prediction,
            "WeightedPooled_RF": pooled_prediction,
            "IrResidualCo_RF": residual_prediction,
        }
        for method, predictions in per_fold_predictions.items():
            fold_rows.append(
                _fold_metric_row(
                    method,
                    fold,
                    co_targets[outer_validation],
                    predictions,
                    ir_train_rows=len(ir_targets)
                    if method != "CoOnly_RF"
                    else 0,
                    co_train_rows=len(outer_train)
                    if method != "IrOnly_ZeroShot_RF"
                    else 0,
                    source_weight=source_weight
                    if method == "WeightedPooled_RF"
                    else None,
                    residual_eta=residual_eta
                    if method == "IrResidualCo_RF"
                    else None,
                )
            )
        print(
            f"  selected Ir weight={source_weight:g}; residual eta={residual_eta:g}",
            flush=True,
        )

    if any(not np.isfinite(values).all() for values in method_predictions.values()):
        raise RuntimeError("At least one method did not produce complete finite Co OOF predictions.")

    summary_rows = []
    for method in METHOD_ORDER:
        metrics = regression_metrics(co_targets, method_predictions[method])
        method_fold_rows = [row for row in fold_rows if row["method"] == method]
        summary_rows.append(
            {
                "dataset": "Co",
                "evaluation": "reaction_grouped_5fold_oof",
                "method": method,
                "ir_train_rows": len(ir_targets) if method != "CoOnly_RF" else 0,
                "co_evaluation_rows": len(co_frame),
                "co_train_rows_per_fold": "0"
                if method == "IrOnly_ZeroShot_RF"
                else "fold-dependent",
                **metrics,
                "fold_r2_mean": float(np.mean([row["r2"] for row in method_fold_rows])),
                "fold_r2_std": float(
                    np.std([row["r2"] for row in method_fold_rows], ddof=1)
                ),
                "fit_select_predict_seconds": method_seconds[method],
            }
        )

    sources = publication_sources(co_frame[NAME_COLUMN].astype(str).tolist())
    prediction_rows = []
    for method in METHOD_ORDER:
        for row_index, (name, source, fold, group, target, prediction) in enumerate(
            zip(
                co_frame[NAME_COLUMN].astype(str),
                sources,
                folds,
                groups,
                co_targets,
                method_predictions[method],
            )
        ):
            prediction_rows.append(
                {
                    "dataset": "Co",
                    "evaluation": "reaction_grouped_5fold_oof",
                    "method": method,
                    "row_index": row_index,
                    "fold": int(fold),
                    "sample_name": name,
                    "publication_source": source,
                    "reaction_group_id": group,
                    "true_delta_g_kcal_mol": target,
                    "predicted_delta_g_kcal_mol": prediction,
                    "residual_predicted_minus_true": prediction - target,
                    "absolute_error": abs(prediction - target),
                    "ir_source_prediction_kcal_mol": source_predictions[row_index],
                    "selected_ir_sample_weight": selected_source_weight[row_index]
                    if method == "WeightedPooled_RF"
                    else None,
                    "selected_residual_eta": selected_residual_eta[row_index]
                    if method == "IrResidualCo_RF"
                    else None,
                    "raw_residual_correction_kcal_mol": correction_component[row_index]
                    if method == "IrResidualCo_RF"
                    else None,
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selection_rows),
    )


def parse_args() -> argparse.Namespace:
    split_dir = PROJECT_DIR / "data" / "ir_holdout"
    parser = argparse.ArgumentParser(
        description="Evaluate leakage-safe Ir-to-Co random-forest transfer baselines."
    )
    parser.add_argument(
        "--ir-development", default=str(split_dir / "Database Ir Development.csv")
    )
    parser.add_argument("--co-data", default=str(PROJECT_DIR / "data" / "Database Co.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_DIR
            / "outputs"
            / "rdkit_rf_transfer"
            / "co_old_seed_10_compact"
        ),
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=str(
            PROJECT_DIR
            / "outputs"
            / "rdkit_ml_baselines"
            / "co_old_seed_10_compact"
            / "feature_cache"
        ),
    )
    parser.add_argument("--source-weight-grid", default="0,0.05,0.20")
    parser.add_argument("--residual-eta-grid", default="0,0.5,1.0")
    parser.add_argument("--fingerprint-bits", type=int, default=REACTION_FINGERPRINT_BITS)
    parser.add_argument(
        "--environment-fingerprint-bits",
        type=int,
        default=ENVIRONMENT_FINGERPRINT_BITS,
    )
    parser.add_argument("--no-3d-descriptors", action="store_true")
    parser.add_argument(
        "--descriptor-num-conformers", type=int, default=DESCRIPTOR_NUM_CONFORMERS
    )
    parser.add_argument(
        "--descriptor-conformer-seed", type=int, default=DESCRIPTOR_CONFORMER_SEED
    )
    parser.add_argument("--no-optimize-conformers", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=ENSEMBLE_TREES)
    parser.add_argument(
        "--selection-estimators", type=int, default=DEFAULT_SELECTION_TREES
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 32 final and 8 selection trees and write into a quick/ subdirectory.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_weight_grid = parse_nonnegative_grid(
        args.source_weight_grid, argument_name="--source-weight-grid"
    )
    residual_eta_grid = parse_nonnegative_grid(
        args.residual_eta_grid, argument_name="--residual-eta-grid"
    )
    if args.fingerprint_bits < 32 or args.environment_fingerprint_bits < 32:
        raise ValueError("Fingerprint widths must both be at least 32.")
    if args.descriptor_num_conformers < 1:
        raise ValueError("--descriptor-num-conformers must be at least 1.")
    if args.native_threads < 1 or args.n_estimators < 1 or args.selection_estimators < 1:
        raise ValueError("Tree counts and --native-threads must be positive.")
    if args.quick:
        args.n_estimators = QUICK_ENSEMBLE_TREES
        args.selection_estimators = QUICK_SELECTION_TREES

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.native_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, str(args.native_threads))

    output_dir = Path(args.output_dir).resolve()
    if args.quick:
        output_dir = output_dir / "quick"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.feature_cache_dir).resolve()
    ir_path = Path(args.ir_development).resolve()
    co_path = Path(args.co_data).resolve()
    ir_frame = _read_table(ir_path)
    co_frame = _read_table(co_path)

    include_3d_descriptors = not args.no_3d_descriptors
    optimize_conformers = not args.no_optimize_conformers
    feature_arguments = {
        "fingerprint_bits": args.fingerprint_bits,
        "environment_fingerprint_bits": args.environment_fingerprint_bits,
        "include_3d_descriptors": include_3d_descriptors,
        "descriptor_num_conformers": args.descriptor_num_conformers,
        "descriptor_conformer_seed": args.descriptor_conformer_seed,
        "optimize_conformers": optimize_conformers,
        "use_cache": not args.no_feature_cache,
    }
    ir_features, names = load_or_build_features(
        ir_frame,
        ir_path,
        cache_dir,
        "ir_development",
        **feature_arguments,
    )
    co_features, co_names = load_or_build_features(
        co_frame,
        co_path,
        cache_dir,
        "co",
        **feature_arguments,
    )
    if names != co_names:
        raise RuntimeError("Ir and Co feature schemas differ.")
    expected_names = feature_names(
        args.fingerprint_bits,
        args.environment_fingerprint_bits,
        include_3d_descriptors,
    )
    if names != expected_names:
        raise RuntimeError("Loaded feature names do not match the requested schema.")

    ir_targets = ir_frame[TARGET_COLUMN].to_numpy(dtype=float)
    co_targets = co_frame[TARGET_COLUMN].to_numpy(dtype=float)
    splits, folds, groups, split_audit = make_co_splits(co_frame, co_targets, args.seed)
    summary, fold_metrics, predictions, selections = evaluate_transfer_methods(
        ir_features,
        ir_targets,
        co_frame,
        co_features,
        splits,
        folds,
        groups,
        source_weight_grid=source_weight_grid,
        residual_eta_grid=residual_eta_grid,
        seed=args.seed,
        n_estimators=args.n_estimators,
        selection_estimators=args.selection_estimators,
        n_jobs=args.n_jobs,
        native_threads=args.native_threads,
    )
    summary = summary.sort_values("r2", ascending=False).reset_index(drop=True)
    from src.regression_classification import export_classification
    predictions = export_classification(
        predictions, output_dir, "Co", group_columns=("method",),
        role="RF transfer: grouped Co OOF; IrOnly method is zero-shot",
        plots=not args.skip_plots,
    )
    summary.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        output_dir / "all_predictions_long.csv", index=False, encoding="utf-8-sig"
    )
    selections.to_csv(
        output_dir / "hyperparameter_selection.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "failures.json").write_text("[]\n", encoding="utf-8")
    (output_dir / "co_cv_split_audit.json").write_text(
        json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_index": np.arange(len(co_frame)),
            "fold": folds,
            "reaction_group_id": groups,
            "source": split_audit["sources"],
            "source_bucket": split_audit["source_buckets"],
            "stratum": split_audit["strata"],
        }
    ).to_csv(output_dir / "co_cv_split_audit.csv", index=False, encoding="utf-8-sig")

    schema = {
        "version": FEATURE_VERSION,
        "feature_dimension": len(names),
        "feature_names": names,
        "rdkit_and_morgan_only": True,
        "matches_main_model_global_rdkit_blocks": False,
        "matches_main_model_global_rdkit_block_layout": True,
        "fingerprint_width_matches_main_model": False,
        "target": "G (kcal/mol), predicted directly",
        "temperature_in_model_inputs": False,
        "xtb_in_model_inputs": False,
        "metal_identity_in_model_inputs": False,
        "role_columns": dict(ROLE_COLUMNS),
        "explicitly_excluded_table_columns": list(FORBIDDEN_MODEL_COLUMNS),
        "reaction_fingerprint_bits": int(args.fingerprint_bits),
        "environment_fingerprint_bits": int(args.environment_fingerprint_bits),
        "fold_local_feature_reduction": {
            "constant_feature_filter": "VarianceThreshold fitted on each training subset",
            "method": "SelectKBest(f_regression) fitted on each training subset",
            "max_selected_features": SELECTED_FEATURES,
        },
        "include_3d_descriptors": include_3d_descriptors,
        "descriptor_num_conformers": int(args.descriptor_num_conformers),
        "descriptor_conformer_seed": int(args.descriptor_conformer_seed),
        "optimize_conformers": optimize_conformers,
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    protocol = {
        "task": "Ir-to-Co random-forest domain adaptation for direct Delta-G prediction",
        "methods": {
            "IrOnly_ZeroShot_RF": "fit only Ir development labels; predict Co directly",
            "CoOnly_RF": "fit only the current Co outer-training fold",
            "WeightedPooled_RF": (
                "fit Ir development plus Co outer-training rows; Co row weight is 1 and "
                "the Ir weight is selected only by grouped inner validation"
            ),
            "IrResidualCo_RF": (
                "Ir source prediction plus eta times a Co-trained residual forest; eta is "
                "selected only by grouped inner validation"
            ),
        },
        "not_neural_checkpoint_finetuning": True,
        "quick_mode": bool(args.quick),
        "quick_mode_is_final_result": False if args.quick else None,
        "outer_evaluation": (
            "five-fold Co OOF; reaction identity groups and source-target balancing are "
            "identical to the current transfer-Co protocol"
        ),
        "inner_selection": (
            "for each outer fold, the other four preassigned group-safe fold blocks are "
            "used as inner validation blocks; the outer validation labels are untouched"
        ),
        "source_weight_grid": list(source_weight_grid),
        "residual_eta_grid": list(residual_eta_grid),
        "final_n_estimators": int(args.n_estimators),
        "selection_n_estimators": int(args.selection_estimators),
        "random_forest_parameters": {
            "max_depth": 10,
            "max_leaf_nodes": 64,
            "max_features": 0.35,
            "min_samples_leaf": 2,
            "max_selected_features": SELECTED_FEATURES,
            "n_jobs": int(args.n_jobs),
            "seed": int(args.seed),
        },
        "input_files": {
            "ir_development": {
                "path": str(ir_path),
                "sha256": _sha256(ir_path),
                "rows": len(ir_frame),
            },
            "co": {
                "path": str(co_path),
                "sha256": _sha256(co_path),
                "rows": len(co_frame),
            },
        },
        "feature_cache_dir": str(cache_dir),
        "feature_schema_file": "feature_schema.json",
        "leakage_guards": [
            "Ir independent test is not read or fitted",
            "Co outer-validation labels are not used for training or transfer-strength selection",
            "reaction-identical Co rows remain in one outer and inner fold block",
            "temperature, xTB and experimental outcome-derived columns are excluded",
        ],
        "domain_identity_note": (
            "the strict model-matched RDKit baseline omits Cat/metal identity; the methods "
            "therefore test structural transfer rather than an explicit Ir/Co domain flag"
        ),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nRandom-forest transfer experiment complete.")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.method}: R2={row.r2:.4f}, RMSE={row.rmse:.4f}, MAE={row.mae:.4f}"
        )
    print(f"Summary: {output_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
