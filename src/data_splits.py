import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np


LEGACY_SPLIT_VERSION = "ir_class_difference_v1"
SPLIT_VERSION = "ir_random_holdout_v1"
RANDOM_ALGORITHM = "numpy.random.RandomState(seed).permutation; first ceil(N*fraction) rows are test; preserve source order"
TARGET_COLUMN = "G (kcal/mol)"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_ids(frame):
    if "Name" not in frame or frame["Name"].isna().any():
        raise ValueError("Every reaction needs a nonempty Name for the holdout split.")
    names = frame["Name"].astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError("Reaction Names must be nonempty and unique within each table.")
    return names


@lru_cache(maxsize=32768)
def _canonical_structure(smiles, marked_indices=""):
    from rdkit import Chem
    from .dataset import parse_reaction_site_indices

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES in holdout audit: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if marked_indices:
        for index in parse_reaction_site_indices(marked_indices, "site", "holdout audit"):
            if index >= mol.GetNumAtoms():
                raise ValueError("Reaction-site index outside molecule in holdout audit")
            mol.GetAtomWithIdx(index).SetAtomMapNum(1)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def model_input_keys(frame):

    keys = []
    for _, row in frame.iterrows():
        keys.append((
            _canonical_structure(row["sub-type"], str(row["bpin_index_sub"])),
            _canonical_structure(row["P"], str(row["bpin_index Production"])),
            _canonical_structure(row["L-type"]),
            _canonical_structure(row["B-type"]),
            _canonical_structure(row["sol"]),
            str(row["Cat"]),
        ))
    return keys


def audit_disjoint_inputs(development, test, reject_model_input_overlap=True):
    development_names, test_names = record_ids(development), record_ids(test)
    overlap = set(development_names) & set(test_names)
    if overlap:
        raise ValueError(f"Training/test Name overlap: {sorted(overlap)[:5]}")
    train_keys, test_keys = model_input_keys(development), model_input_keys(test)
    shared = set(train_keys) & set(test_keys)
    overlapping_test_names = test_names[[key in shared for key in test_keys]].tolist()
    if shared and reject_model_input_overlap:
        names = test_names[[key in shared for key in test_keys]].tolist()
        raise ValueError(f"Training/test canonical model-input overlap: {names[:5]}")
    component_overlap = {}
    for column in ("sub-type", "P", "L-type"):
        known = set(development[column].map(_canonical_structure))
        values = test[column].map(_canonical_structure)
        component_overlap[column] = {
            "test_rows_seen_in_development": int(values.isin(known).sum()),
            "test_unique_structures": int(values.nunique()),
        }
    return {
        "record_overlap": 0,
        "canonical_model_input_overlap": len(shared),
        "overlapping_model_input_test_names": overlapping_test_names,
        "overlapping_model_input_test_rows": len(overlapping_test_names),
        "test_repeated_model_input_rows": len(test_keys)-len(set(test_keys)),
        "component_overlap": component_overlap,
    }


def derive_ir_test_set(full, development):

    full_names, development_names = record_ids(full), record_ids(development)
    absent = set(development_names)-set(full_names)
    if absent:
        raise ValueError(f"Class contains records missing from Database Ir: {sorted(absent)[:5]}")
    mask = ~full_names.isin(development_names)
    test = full.loc[mask].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError("No independent Ir test records remain.")
    common = [column for column in full.columns if column in development and column != "Name"]
    left = full.copy()
    right = development.copy()
    left["Name"], right["Name"] = full_names, development_names
    shared = left.merge(right, on="Name", suffixes=("_full", "_class"), validate="one_to_one")
    differences = []
    for column in common:
        a, b = shared[column+"_full"], shared[column+"_class"]
        different = ~(a.eq(b) | (a.isna() & b.isna()))
        for _, row in shared.loc[different].iterrows():
            differences.append({"Name": str(row["Name"]), "column": column,
                                "full_value": str(row[column+"_full"]),
                                "class_value": str(row[column+"_class"])})
    audit = {
        "split_version": LEGACY_SPLIT_VERSION,
        "selection_rule": "Database Ir records whose unique Name is absent from Database Ir Class",
        "full_rows": len(full), "development_rows": len(development), "test_rows": len(test),
        "shared_record_count": len(shared),
        "shared_value_differences": differences,
        "test_names": record_ids(test).tolist(),
        "test_source_row_indices_zero_based": [int(i) for i in full.index[mask]],
        "test_used_for_model_selection": False,
        "evaluation_rule": "select one checkpoint by Class validation R2/RMSE, then evaluate holdout once",
        "audit": audit_disjoint_inputs(development, test),
        "limitation": "record-level holdout from the same source; not an unseen-substrate/ligand or Co benchmark",
    }
    return test, audit


def load_frozen_ir_split(development_path, test_path, manifest_path, full_path=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("split_version") not in (SPLIT_VERSION, LEGACY_SPLIT_VERSION):
        raise ValueError("Unsupported Ir split manifest; rerun scripts/prepare_ir_test_set.py.")
    for name, path in (("development", development_path), ("test", test_path)):
        if sha256_file(path) != manifest["sources"][name]["sha256"]:
            raise ValueError(f"{name} data changed since the frozen split. Rerun scripts/prepare_ir_test_set.py and retrain.")
    development, test = pd.read_csv(development_path), pd.read_csv(test_path)
    if len(development) != manifest["development_rows"] or len(test) != manifest["test_rows"]:
        raise ValueError("Frozen split row counts do not match the manifest.")
    if record_ids(test).tolist() != manifest["test_names"]:
        raise ValueError("Frozen test identities/order do not match the manifest.")
    random_split = manifest["split_version"] == SPLIT_VERSION
    if random_split:
        if full_path is None or sha256_file(full_path) != manifest["sources"]["full"]["sha256"]:
            raise ValueError("Full Ir source missing or changed since the random split; use a new split/run.")
        full = pd.read_csv(full_path)
        development_indices, test_indices = _random_split_indices(
            len(full), manifest["test_fraction"], manifest["test_split_seed"])
        if (manifest["random_algorithm"] != RANDOM_ALGORITHM
                or manifest["development_source_row_indices_zero_based"] != development_indices.tolist()
                or manifest["test_source_row_indices_zero_based"] != test_indices.tolist()):
            raise ValueError("Saved rows do not reproduce the configured random seed and test fraction.")
        for actual, indices in ((development, development_indices), (test, test_indices)):
            pd.testing.assert_frame_equal(actual, full.iloc[indices].reset_index(drop=True),
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
    audit_disjoint_inputs(development, test, reject_model_input_overlap=not random_split)
    return development, test, manifest


def checkpoint_split_config(manifest):
    config = {
        "split_version": manifest["split_version"],
        "development_sha256": manifest["sources"]["development"]["sha256"],
        "test_sha256": manifest["sources"]["test"]["sha256"],
        "development_rows": manifest["development_rows"],
        "test_rows": manifest["test_rows"],
        "test_used_for_model_selection": False,
        "evaluation_rule": manifest["evaluation_rule"],
    }
    if manifest["split_version"] == SPLIT_VERSION:
        config.update(full_sha256=manifest["sources"]["full"]["sha256"],
                      test_split_seed=manifest["test_split_seed"],
                      test_fraction=manifest["test_fraction"],
                      random_algorithm=manifest["random_algorithm"], split_unit="record")
    return config


def _random_split_indices(row_count, test_fraction, seed):
    if not 0 < float(test_fraction) < 1:
        raise ValueError("test_fraction must be between zero and one.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError("The holdout seed must be an integer between 0 and 2**32-1.")
    count = math.ceil(row_count * float(test_fraction))
    if count < 1 or row_count-count < 5:
        raise ValueError("Need a nonempty test set and at least five development rows.")
    permutation = np.random.RandomState(int(seed)).permutation(row_count)
    return np.sort(permutation[count:]), np.sort(permutation[:count])


def derive_random_ir_split(full, seed=10, test_fraction=0.10):

    record_ids(full)
    development_indices, test_indices = _random_split_indices(len(full), test_fraction, seed)
    development = full.iloc[development_indices].copy().reset_index(drop=True)
    test = full.iloc[test_indices].copy().reset_index(drop=True)
    audit = {
        "split_version": SPLIT_VERSION, "test_split_seed": int(seed),
        "test_fraction": float(test_fraction), "realized_test_fraction": len(test)/len(full),
        "random_algorithm": RANDOM_ALGORITHM, "split_unit": "record", "test_stratified": False,
        "selection_rule": "Uniform random sample without replacement from all Database Ir records",
        "full_rows": len(full), "development_rows": len(development), "test_rows": len(test),
        "development_names": record_ids(development).tolist(), "test_names": record_ids(test).tolist(),
        "development_source_row_indices_zero_based": development_indices.tolist(),
        "test_source_row_indices_zero_based": test_indices.tolist(),
        "test_used_for_model_selection": False,
        "evaluation_rule": "select one checkpoint by development CV validation R2/RMSE, then evaluate holdout once",
        "audit": audit_disjoint_inputs(development, test, reject_model_input_overlap=False),
        "limitation": "random record holdout; related or identical model inputs may cross the split; not grouped chemical generalization",
    }
    return development, test, audit


def ensure_random_ir_split(full_path, development_path, test_path, manifest_path,
                           seed=10, test_fraction=0.10):

    manifest_path = Path(manifest_path)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("split_version") != SPLIT_VERSION
                or manifest.get("test_split_seed") != int(seed)
                or manifest.get("test_fraction") != float(test_fraction)):
            raise ValueError("Existing split uses a different protocol; use a separate seed/fraction directory.")
        return load_frozen_ir_split(development_path, test_path, manifest_path, full_path=full_path)
    if Path(development_path).exists() or Path(test_path).exists():
        raise FileExistsError("Incomplete split files already exist without a manifest; inspect them before regenerating.")
    full = pd.read_csv(full_path)
    development, test, manifest = derive_random_ir_split(full, seed, test_fraction)
    for frame, path in ((development, development_path), (test, test_path)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    manifest["sources"] = {
        name: {"file_name": Path(path).name, "sha256": sha256_file(path)}
        for name, path in (("full", full_path), ("development", development_path), ("test", test_path))
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return development, test, manifest
