from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .protocol import ROOT, TARGET, PREDICTED, dump, sha256, verify_split, write_results
from baselines.ML import train_rdkit_baselines as ml
from baselines.ML import train_random_forest_transfer as rf


def cached_features(frame, source, key, work_dir):


    cache_dir = ROOT / "outputs/baselines/feature_cache"
    expected = ml._cache_key(Path(source), 256, 128, True, 3, ml.DESCRIPTOR_CONFORMER_SEED, True)
    npz = cache_dir / f"{key}_features.npz"
    meta = npz.with_suffix(".json")
    if npz.is_file() and meta.is_file():
        saved = json.loads(meta.read_text(encoding="utf-8"))

        if {k: v for k, v in saved.items() if k != "data_path"} != {k: v for k, v in expected.items() if k != "data_path"}:
            raise ValueError(f"Stale or incompatible feature cache: {npz}")
        with np.load(npz, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
        provenance = {"cache": str(npz), "sha256": sha256(npz), "metadata_sha256": sha256(meta)}
    else:
        features, _ = ml.load_or_build_features(frame, Path(source), Path(work_dir) / "feature_cache", key,
                                                256, 128, True, 3, ml.DESCRIPTOR_CONFORMER_SEED, True, True)
        provenance = {"cache": str(Path(work_dir) / "feature_cache" / f"{key}_features.npz"), "rebuilt": True}
    if features.shape != (len(frame), 1302) or not np.isfinite(features).all():
        raise ValueError("Feature cache shape or values are invalid.")
    return features, provenance


def save_vector(frame, values, output, model, evaluation, plots, folds=None):
    predictions = pd.DataFrame({"Name": frame.Name.to_numpy(), PREDICTED: np.asarray(values).reshape(-1)})
    if folds is not None:
        predictions["Fold"] = folds
    return write_results(predictions, frame.Name.tolist(), frame[TARGET].to_numpy(),
                         Path(output) / "evaluations" / model / evaluation,
                         model, evaluation, plots=plots)


def fit_standard(development, test, x_development, x_test, splits, folds, registry,
                 directory, output, native_threads=1, plots=True):
    y = development[TARGET].to_numpy(float)
    for name, template in registry.items():
        print(f"[holdout baseline] {name}: development CV, then fit all {len(y)} development rows", flush=True)
        oof = np.full(len(y), np.nan)
        for train, validation in splits:
            fitted = clone(template)
            with ml.bounded_native_threads(native_threads):
                fitted.fit(x_development[train], y[train])
                oof[validation] = ml._predict_vector(fitted, x_development[validation])
        save_vector(development, oof, output, name, "development_oof", plots, folds)
        final = clone(template)
        with ml.bounded_native_threads(native_threads):
            final.fit(x_development, y)

        joblib.dump(final, Path(directory) / f"{name}.joblib")
        values = ml._predict_vector(final, x_test)
        save_vector(test, values, output, name, "holdout_test", plots)


def fit_rf_transfer(development, test, x_development, x_test, ir_x, ir_y,
                    splits, folds, groups, directory, output, seed=10,
                    trees=160, selection_trees=40, native_threads=1, plots=True):
    y = development[TARGET].to_numpy(float)
    source_grid, eta_grid = rf.DEFAULT_SOURCE_WEIGHT_GRID, rf.DEFAULT_RESIDUAL_ETA_GRID

    _, _, oof, selection_rows = rf.evaluate_transfer_methods(
        ir_x, ir_y, development, x_development, splits, folds, groups,
        source_weight_grid=source_grid, residual_eta_grid=eta_grid, seed=seed,
        n_estimators=trees, selection_estimators=selection_trees, n_jobs=1, native_threads=native_threads)
    selection_rows.to_csv(Path(directory) / "development_nested_selections.csv", index=False)
    for name, frame in oof.groupby("method"):
        if name == "CoOnly_RF":
            continue
        evaluation = "development_zero_shot" if name == "IrOnly_ZeroShot_RF" else "development_oof"
        write_results(frame, development.Name.tolist(), y,
                      Path(output) / "evaluations" / name / evaluation, name, evaluation, plots)


    inner = rf.inner_splits_from_outer_assignment(np.arange(len(y)), folds, groups)
    source = rf._fit_forest(ir_x, ir_y, seed=seed, n_estimators=trees, n_jobs=1, native_threads=native_threads)
    source_development = source.predict(x_development)
    common = dict(inner_splits=inner, outer_fold=0, seed=seed,
                  n_estimators=selection_trees, n_jobs=1, native_threads=native_threads)
    weight, weight_rows = rf.select_source_weight(ir_x, ir_y, x_development, y, candidates=source_grid, **common)
    eta, eta_rows = rf.select_residual_eta(source_development, x_development, y, candidates=eta_grid, **common)
    pd.DataFrame(weight_rows + eta_rows).to_csv(Path(directory) / "final_development_selections.csv", index=False)
    pooled = rf._fit_weighted_pooled_forest(ir_x, ir_y, x_development, y, source_weight=weight,
                                          seed=seed, n_estimators=trees, n_jobs=1, native_threads=native_threads)
    residual = rf._fit_forest(x_development, y-source_development, seed=seed, n_estimators=trees,
                              n_jobs=1, native_threads=native_threads)
    bundle = {"source": source, "pooled": pooled, "residual": residual, "weight": weight, "eta": eta}
    joblib.dump(bundle, Path(directory) / "rf_transfer_models.joblib")
    dump(Path(directory) / "final_selection.json", {"source_weight": weight, "residual_eta": eta,
                                                  "selection_rows": len(y), "test_labels_used": False})
    source_test = source.predict(x_test)
    for name, values in (("IrOnly_ZeroShot_RF", source_test),
                         ("WeightedPooled_RF", pooled.predict(x_test)),
                         ("IrResidualCo_RF", source_test + eta * residual.predict(x_test))):
        evaluation = "holdout_zero_shot" if name == "IrOnly_ZeroShot_RF" else "holdout_test"
        save_vector(test, values, output, name, evaluation, plots)


def run_baselines(args, manifest):
    output = Path(args.output_dir)
    verify_split(output, manifest)
    directory = output / "training/classical"
    directory.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(output / "split/development.csv")
    test = pd.read_csv(output / "split/test.csv")
    full = pd.read_csv(manifest["source"])
    x, co_provenance = cached_features(full, manifest["source"], "co", directory)
    x_development, x_test = x[manifest["development_indices"]], x[manifest["test_indices"]]
    audit = pd.read_csv(output / "split/development_folds.csv")
    np.testing.assert_array_equal(audit.Name.astype(str), development.Name.astype(str))
    folds = audit.fold.to_numpy(int)
    groups = audit.reaction_group_id.to_numpy()
    splits = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(1, 6)]
    registry = ml.build_model_registry(seed=args.cv_seed, n_jobs=1, quick=args.smoke)
    fit_standard(development, test, x_development, x_test, splits, folds, registry,
                 directory, output, args.native_threads, not args.skip_plots)
    ir_path = ROOT / "data/ir_holdout/Database Ir Development.csv"
    ir = pd.read_csv(ir_path)
    ir_x, ir_provenance = cached_features(ir, ir_path, "ir_development", directory)
    dump(directory / "feature_provenance.json", {"Co": co_provenance, "Ir": ir_provenance,
                                               "Ir_source_sha256": sha256(ir_path), "target_rows_used_for_fitting": len(development)})
    fit_rf_transfer(development, test, x_development, x_test, ir_x, ir[TARGET].to_numpy(),
                    splits, folds, groups, directory, output, args.cv_seed,
                    32 if args.smoke else 160, 8 if args.smoke else 40, args.native_threads, not args.skip_plots)
