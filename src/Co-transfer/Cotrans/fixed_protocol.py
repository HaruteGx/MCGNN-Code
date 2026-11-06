from pathlib import Path

import numpy as np
import pandas as pd

from co_transfer_holdout.protocol import dump, read_json, sha256, verify_split, write_results
from experiment_protocol import validate_tables, verified_copy

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "config/co_reference"
SPLIT_FILES = ("development.csv", "test.csv", "development_folds.csv")


def reference_tables():
    manifest = read_json(REFERENCE / "manifest.json")
    expected = {"seed": 126, "cv_seed": 10, "development_size": 125,
                "test_size": 14, "class_boundaries": [24., 26.]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Co reference protocol changed")
    for name in SPLIT_FILES:
        if sha256(REFERENCE / name) != manifest["files"][name]:
            raise ValueError(f"Reference split file changed: {name}")
    development, test, audit = [pd.read_csv(REFERENCE / name) for name in SPLIT_FILES]
    validate_tables(development, test, audit)
    if manifest["group_overlap"]:
        raise ValueError("Development/test reaction groups overlap")
    return manifest, development, test, audit


def validate_source_data(path):

    manifest, development, test, _ = reference_tables()
    expected = pd.concat([
        development.set_axis(manifest["development_indices"]),
        test.set_axis(manifest["test_indices"]),
    ]).sort_index()
    if expected.index.tolist() != list(range(139)):
        raise ValueError("Reference row indices do not cover 139 records")
    frame = pd.read_csv(path)
    try:
        pd.testing.assert_frame_equal(frame, expected, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("Use the original 139-record Database Co table restored from seed-126 inputs") from error
    return frame


def prepare_frozen_split(output, data_path):
    output, data_path = Path(output).resolve(), Path(data_path).resolve()
    allowed = [ROOT / "outputs/co_transfer", ROOT / "outputs/baselines/co_transfer_holdout"]
    if not any(root.resolve() in output.parents for root in allowed):
        raise ValueError("Choose an isolated Co output run directory")
    protected = [REFERENCE.resolve(), (ROOT / "outputs/co_transfer/co_new_v1").resolve()]
    if any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError("Choose a new run directory for output")
    validate_source_data(data_path)
    historical, _, _, _ = reference_tables()
    manifest = dict(historical)
    manifest.update(source=str(data_path), source_sha256=sha256(data_path),
                    version="co_fixed_historical_seed_126_v1", mode="holdout",
                    reference_manifest_sha256=sha256(REFERENCE / "manifest.json"),
                    historical_source_sha256=historical.get("historical_source_sha256", historical["source_sha256"]),
                    sampling="Exact saved seed-126 split and CV-seed-10 folds; no search or redraw",
                    historical_split_was_search_selected=True)
    destination = output / "split/manifest.json"
    if destination.exists() and read_json(destination) != manifest:
        raise ValueError("Existing split differs; choose a new output directory")
    for name in SPLIT_FILES:
        verified_copy(REFERENCE / name, output / "split" / name)
    if not destination.exists():
        dump(destination, manifest)
    return verify_split(output)


def prepare_inputs(args):
    from co_transfer_holdout.run_co_holdout import validate_ir_source
    info = validate_ir_source(args.checkpoint)
    expected = read_json(REFERENCE / "frozen_inputs.json")["source_checkpoint_sha256"]
    if info["sha256"] != expected:
        raise ValueError("Use the configured Ir-pretrained source checkpoint")
    prepare_frozen_split(args.output_dir, args.data)
    source_hash = verified_copy(args.checkpoint, args.output_dir / "inputs/ir_source_checkpoint.pth")
    inputs = dict(source_checkpoint_sha256=source_hash,
                  split_manifest_sha256=sha256(args.output_dir / "split/manifest.json"),
                  graph_cache_key=args.cache_key, source_data_sha256=sha256(args.data))
    path = args.output_dir / "frozen_inputs.json"
    if path.exists() and read_json(path) != inputs:
        raise ValueError("Frozen source/cache identity changed; choose a new run directory")
    if not path.exists():
        dump(path, inputs)
    return tuple(pd.read_csv(args.output_dir / "split" / name) for name in SPLIT_FILES)

