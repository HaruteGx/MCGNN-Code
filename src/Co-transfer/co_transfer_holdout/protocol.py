from __future__ import annotations


import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from src.regression_classification import BOUNDARIES, attach_classification

TARGET = "G (kcal/mol)"
TRUE = "True G (kcal/mol)"
PREDICTED = "Predicted G (kcal/mol)"
IR_CHECKPOINT = ROOT / "models/Ir checkpoints.pth"
CO_CHECKPOINT = ROOT / "models/Co transfer checkpoints.pth"
VERSION = "co_random_14_holdout_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def isolated_output(path):

    resolved = Path(path).resolve()
    allowed = (ROOT / "outputs/baselines/co_transfer_holdout").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"Choose a run subdirectory below {allowed}")
    return resolved


def derive_split(frame, seed=10, test_size=14):
    if not 1 <= test_size < len(frame) - 5:
        raise ValueError("Need at least six development records after splitting.")
    if frame.Name.isna().any() or not frame.Name.is_unique:
        raise ValueError("Co Name values must be unique and non-missing.")
    if not np.isfinite(frame[TARGET].to_numpy(float)).all():
        raise ValueError("Co experimental labels must be finite.")

    test_indices = np.sort(np.random.RandomState(seed).permutation(len(frame))[:test_size])
    development_indices = np.flatnonzero(~np.isin(np.arange(len(frame)), test_indices))
    return development_indices, test_indices


def prepare(output, data_path, seed=10, test_size=14, cv_seed=10):
    from baselines.ML.train_rdkit_baselines import make_co_splits, reaction_group_ids

    output, data_path = Path(output), Path(data_path).resolve()
    directory = output / "split"
    manifest_path = directory / "manifest.json"
    frame = pd.read_csv(data_path)
    if len(frame) != 139 or test_size != 14:
        raise ValueError("This experiment requires the current 139-record Co table and exactly 14 test records.")
    development_indices, test_indices = derive_split(frame, seed, test_size)
    expected = {"version": VERSION, "source_sha256": sha256(data_path), "seed": seed,
                "test_size": test_size, "cv_seed": cv_seed}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen split differs from current data/settings; use a new isolated run directory.")
        verify_split(output, manifest)
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("An incomplete split directory exists; use a new isolated run directory.")
    groups = reaction_group_ids(frame)
    overlap = sorted(set(groups[development_indices]) & set(groups[test_indices]))
    if overlap:
        raise ValueError(f"Random split crosses reaction identities: {overlap}. No automatic redraw is performed.")
    development, test = frame.iloc[development_indices].copy(), frame.iloc[test_indices].copy()

    cv_targets = development[TARGET].to_numpy(np.float32).astype(float)
    _, folds, development_groups, audit = make_co_splits(development, cv_targets, seed=cv_seed)
    directory.mkdir(parents=True, exist_ok=True)
    development.to_csv((directory / "development.csv"), index=False, encoding="utf-8-sig")
    test.to_csv((directory / "test.csv"), index=False, encoding="utf-8-sig")
    audit_table = pd.DataFrame({"Name": development.Name.to_numpy(), "fold": folds,
                                "reaction_group_id": development_groups})
    audit_table.to_csv((directory / "development_folds.csv"), index=False)
    names = {"development": development.Name.astype(str).tolist(), "test": test.Name.astype(str).tolist()}
    manifest = {**expected, "source": str(data_path), "created_utc": datetime.now(timezone.utc).isoformat(),
                "development_size": len(development), "development_indices": development_indices.tolist(),
                "test_indices": test_indices.tolist(), "names": names,
                "group_overlap": overlap, "development_cv_audit": audit,
                "sampling": "RandomStateseed.permutation(N)[:14], retaining original CSV row order",
                "test_role": "Internal record holdout",
                "class_boundaries": list(BOUNDARIES["Co"]),
                "files": {name: sha256(directory / name) for name in ("development.csv", "test.csv", "development_folds.csv")}}
    dump(manifest_path, manifest)
    return manifest


def verify_split(output, manifest=None):
    directory = Path(output) / "split"
    manifest = manifest or read_json(directory / "manifest.json")
    if sha256(manifest["source"]) != manifest["source_sha256"]:
        raise ValueError("Original Co data changed since the holdout was frozen.")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Frozen split file was modified: {name}")
    return manifest


def snapshot_current(output):

    target = Path(output) / "original_results_snapshot"
    receipt = target / "snapshot_manifest.json"
    if receipt.exists():
        info = read_json(receipt)
        for record in info["files"]:
            if sha256(target / record["relative_path"]) != record["sha256"]:
                raise ValueError("Preservation snapshot was modified.")
        return info
    selected = {IR_CHECKPOINT, CO_CHECKPOINT, ROOT / "data/Database Co.csv"}
    for relative in ("outputs/ir_prediction", "outputs/01_dataset_distribution/figures", "outputs/ir_prediction/figures", "outputs/interpretability/figures", "outputs/co_transfer/figures",
                     "outputs/Model_transferCo_Regress_Results", "outputs/baselines_Co", "outputs/baselines",
                     "outputs/interpretability/graph_ablations", "outputs/interpretability/graph_pair_ablations"):
        directory = ROOT / relative
        for path in directory.rglob("*"):
            if not path.is_file() or any(x.startswith(".") or x == "feature_cache" for x in path.relative_to(directory).parts):
                continue
            if path.suffix.lower() in {".csv", ".json", ".md", ".png", ".svg", ".pdf", ".xlsx"}:
                selected.add(path)
    records = []
    for source in sorted(selected):
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(ROOT)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256(source)
        if sha256(destination) != digest:
            raise RuntimeError(f"Snapshot verification failed: {relative}")
        records.append({"relative_path": relative.as_posix(), "sha256": digest, "bytes": source.stat().st_size})
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": records,
            "scope": "Saved numerical results, reports, publication figures and stable Ir/Co checkpoints; original files retained"}
    dump(receipt, info)
    return info


def aligned_predictions(frame, names, truth):

    frame = frame.rename(columns={"sample_name": "Name", "Sample Name": "Name",
                                  "true_delta_g_kcal_mol": TRUE, "predicted_delta_g_kcal_mol": PREDICTED}).copy()
    if frame.Name.isna().any() or not frame.Name.is_unique or set(frame.Name.astype(str)) != set(map(str, names)):
        raise ValueError("Prediction rows do not match the complete frozen evaluation set.")
    frame["Name"] = frame.Name.astype(str)
    frame = frame.set_index("Name").loc[list(map(str, names))].reset_index()
    if "status" in frame and not frame.status.eq("OK").all():
        raise ValueError("At least one prediction failed; no reduced-denominator accuracy is reported.")
    if not np.isfinite(frame[PREDICTED].to_numpy(float)).all():
        raise ValueError("Non-finite prediction.")
    if TRUE in frame and frame[TRUE].notna().any():
        np.testing.assert_allclose(frame[TRUE], truth, atol=2e-6, rtol=0)
    frame[TRUE] = np.asarray(truth, dtype=float)
    frame["status"] = "OK"
    return attach_classification(frame, "Co")


def write_results(frame, names, truth, directory, model, evaluation, plots=True):
    from src.regression_plotting import interval_metrics, prediction_report

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scored = aligned_predictions(frame, names, truth)
    scored["model"], scored["evaluation"] = model, evaluation
    scores = interval_metrics(scored[TRUE].to_numpy(), scored[PREDICTED].to_numpy(), BOUNDARIES["Co"])

    n, proportion, z = scores["n"], scores["accuracy"], 1.959963984540054
    center = (proportion + z*z/(2*n)) / (1 + z*z/n)
    half = z*math.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n)) / (1 + z*z/n)
    scores.update({"model": model, "evaluation": evaluation,
                   "accuracy_wilson_95_low": center-half, "accuracy_wilson_95_high": center+half})
    scored.to_csv((directory / "predictions.csv"), index=False, encoding="utf-8-sig")
    dump(directory / "metrics.json", scores)
    if plots:
        prediction_report(scored, directory / "figures", boundaries=BOUNDARIES["Co"],
                          title=f"{model}: {evaluation}", role_note="Co holdout protocol; fixed 24 / 26 kcal/mol boundaries")
    return scored, scores


def combine_reports(output):
    output = Path(output)
    tables, metrics = [], []
    for path in sorted((output / "evaluations").glob("*/*/metrics.json")):
        row = read_json(path)
        metrics.append({k: v for k, v in row.items() if not isinstance(v, (list, dict))})
        tables.append(pd.read_csv((path.with_name("predictions.csv"))))
    if not tables:
        return None
    summary = pd.DataFrame(metrics)
    summary.to_csv((output / "summary_metrics.csv"), index=False)
    pd.concat(tables, ignore_index=True).to_csv((output / "all_predictions.csv"), index=False)
    summary.loc[summary.evaluation.isin(["holdout_test", "holdout_zero_shot"])].sort_values(
        ["evaluation", "accuracy"], ascending=[True, False]).to_csv((output / "test_comparison.csv"), index=False)
    return summary
