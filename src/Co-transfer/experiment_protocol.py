from __future__ import annotations

import shutil
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


from co_transfer_holdout.protocol import dump, read_json, sha256, verify_split

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/co_transfer"


def canonical_settings(settings):

    path_fields = {"reference_run", "output_dir", "xtb_root"}
    def normalize(value, key=None):
        if isinstance(value, dict):
            if key == "code_hashes":
                result = {}
                for path, digest in value.items():
                    name = os.path.normcase(os.path.normpath(path))
                    if name in result and result[name] != digest:
                        raise ValueError(f"Conflicting code hashes for {path}")
                    result[name] = digest
                return result
            return {k: normalize(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if key in path_fields and isinstance(value, str):
            return os.path.normcase(os.path.normpath(str(Path(value).resolve())))
        return value
    return normalize(settings)


def settings_differences(previous, current):
    differences = []
    def visit(before, after, prefix=""):
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                visit(before.get(key, "<missing>"), after.get(key, "<missing>"), f"{prefix}.{key}" if prefix else key)
        elif before != after:
            differences.append(f"{prefix}: {before!r} -> {after!r}")
    visit(canonical_settings(previous), canonical_settings(current))
    return differences


def register_run_settings(output, settings):

    output = Path(output)
    path = output / "run_settings.json"
    if not path.exists():
        dump(path, settings)
        return "created"
    previous = read_json(path)
    changes = settings_differences(previous, settings)
    if not changes:

        return "unchanged"
    preparation_only = {"inputs", "split", ".matplotlib", "frozen_inputs.json", "run_settings.json", "preparation_history"}
    artifacts = [item.name for item in output.iterdir() if item.name not in preparation_only]
    if artifacts:
        details = "\n".join(changes[:12])
        raise ValueError("Run settings/code changed after training or graph preparation started. "
                         "Use a new --output-dir under outputs/co_transfer; --resume cannot mix settings.\n"
                         f"Existing artifacts: {', '.join(sorted(artifacts))}\nChanged fields:\n{details}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dump(output / "preparation_history" / f"{stamp}.json", {"previous_settings": previous, "changes": changes})
    dump(path, settings)
    print("Preparation settings updated.\n" + "\n".join(changes[:12]), flush=True)
    return "updated_preparation"


def output_path(path, smoke=False):
    result = Path(path).resolve()
    if result == OUTPUT_ROOT.resolve() or not result.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Output must be a run subdirectory inside outputs/co_transfer")
    if smoke != result.name.startswith("smoke_"):
        raise ValueError("Smoke output names must start with smoke_; formal output names must not")
    return result


def verified_copy(source, destination):
    source, destination = Path(source), Path(destination)
    expected = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(f"Frozen input was changed: {destination}")
    else:
        shutil.copy2(source, destination)
    if sha256(destination) != expected:
        raise IOError(f"Copy verification failed: {destination}")
    return expected


def validate_tables(development, test, folds):
    for data, expected in ((development, 125), (test, 14)):
        if len(data) != expected or data.Name.isna().any() or not data.Name.is_unique:
            raise ValueError("Expected unique complete 125/14 sample tables")
    if set(development.Name) & set(test.Name):
        raise ValueError("Development and test overlap")
    if not folds.Name.is_unique or set(folds.Name) != set(development.Name):
        raise ValueError("Fold/sample membership mismatch")
    aligned = folds.set_index("Name").loc[development.Name]
    if set(aligned.fold) != {1, 2, 3, 4, 5} or aligned.reaction_group_id.isna().any():
        raise ValueError("Expected complete five-fold audit")
    if aligned.groupby("reaction_group_id").fold.nunique().gt(1).any():
        raise ValueError("Reaction identity crosses validation folds")
    return aligned.fold.to_numpy(int)


def prepare_inputs(args):
    from co_transfer_holdout.run_co_holdout import validate_ir_source
    original = args.reference_run.resolve()
    manifest = verify_split(original)
    if manifest["development_size"] != 125 or manifest["test_size"] != 14 or manifest["seed"] != 10:
        raise ValueError("Use the existing seed-10 125/14 experiment")
    if manifest["class_boundaries"] != [24.0, 26.0]:
        raise ValueError("Co classification boundaries must remain 24/26")
    output = output_path(args.output_dir, args.smoke)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("development.csv", "test.csv", "development_folds.csv", "manifest.json"):
        verified_copy(original / "split" / name, output / "split" / name)
    verify_split(output)
    development = pd.read_csv(output / "split/development.csv")
    test = pd.read_csv(output / "split/test.csv")
    folds = pd.read_csv(output / "split/development_folds.csv")
    validate_tables(development, test, folds)
    source = original / "inputs/ir_source_checkpoint.pth"
    info = validate_ir_source(source)
    expected = read_json(original / "run_settings.json")["source_checkpoint_sha256"]
    if info["sha256"] != expected:
        raise ValueError("Reference experiment Ir source was modified")
    source_hash = verified_copy(source, output / "inputs/ir_source_checkpoint.pth")
    inputs = {"split_manifest_sha256": sha256(output / "split/manifest.json"), "source_checkpoint_sha256": source_hash,
              "xtb_feature_csv_sha256": sha256(args.xtb_root / "xtb_unique_molecule_features.csv"),
              "reference_run": str(original), "xtb_root": str(args.xtb_root.resolve()),
              "boundaries": [24.0, 26.0], "sampling": "Exact existing split and folds; no redraw"}
    path = output / "frozen_inputs.json"
    if path.exists() and canonical_settings(read_json(path)) != canonical_settings(inputs):
        raise ValueError("Frozen experiment inputs changed; use a new output directory")
    if not path.exists():
        dump(path, inputs)
    return development, test, folds


def unlabelled_test(data):
    forbidden = [c for c in data if c == "G (kcal/mol)" or c.startswith(("True ", "Predicted ", "Class "))]
    return data.drop(columns=forbidden).copy()


def assert_checkpoint_protocol(metadata, output, fold):
    development = pd.read_csv(output / "split/development.csv")
    audit = pd.read_csv(output / "split/development_folds.csv").set_index("Name").loc[development.Name]
    valid = np.flatnonzero(audit.fold.to_numpy() == fold)
    train = np.flatnonzero(audit.fold.to_numpy() != fold)
    np.testing.assert_array_equal(metadata["train_indices"], train)
    np.testing.assert_array_equal(metadata["validation_indices"], valid)
    if metadata.get("task") != "co_transfer_fixed_boundary_ordinal_classification" or metadata.get("cv_fold") != fold:
        raise ValueError("This is not a classification-trained fold checkpoint")
    if metadata.get("split_manifest_sha256") != sha256(output / "split/manifest.json"):
        raise ValueError("Checkpoint belongs to a different split")
    if Path(metadata.get("data_path", "")).resolve() != (output / "split/development.csv").resolve():
        raise ValueError("Checkpoint development table differs")


def write_receipt(directory, names, metadata=None):
    dump(directory / "complete.json", {"files": {name: sha256(directory / name) for name in names}, "metadata": metadata or {}})


def verify_receipt(directory):
    receipt = read_json(directory / "complete.json")
    for name, digest in receipt["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Completed artifact changed: {directory / name}")
    return receipt
