from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import rdBase
from co_transfer_holdout.protocol import dump, read_json, sha256

ROOT = Path(__file__).resolve().parents[3]
INPUT_COLUMNS = ["Name", "sol", "sub-type", "L-type", "B-type", "P", "Cat", "T / C",
                 "bpin_index_sub", "bpin_index Production"]


def cache_identity(frame, settings, xtb_root):
    from src.xtb_features import XTBFeatureStore
    source = frame[INPUT_COLUMNS].copy()

    fingerprint = hashlib.sha256(source.to_json(orient="split", index=False, double_precision=15).encode()).hexdigest()
    files = [ROOT / "src" / name for name in
             ("dataset.py", "graph_builders.py", "features.py", "descriptors.py", "mcencoding.py", "xtb_features.py")]
    xtb_files = {}
    if settings["use_xtb_features"]:
        store = XTBFeatureStore(xtb_root)
        for _, row in store.jobs.values():
            job = Path(xtb_root) / "calculations" / row["role"] / row["job_id"]
            for name in ("xtbout.json", "result.json", "xtbopt.xyz", "sp.out", "wbo", "vibspectrum"):
                path = job / name
                xtb_files[str(path.relative_to(xtb_root))] = sha256(path) if path.exists() else "missing"
        xtb_files["xtb_unique_molecule_features.csv"] = sha256(Path(xtb_root) / "xtb_unique_molecule_features.csv")
    identity = dict(version=1, inputs_sha256=fingerprint, feature_settings=settings,
                    code_hashes={p.name: sha256(p) for p in files}, cache_code_sha256=sha256(Path(__file__)),
                    xtb_sources=xtb_files, rdkit=rdBase.rdkitVersion, torch=torch.__version__, dgl=dgl.__version__)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key, identity


def build_or_load(frame, settings, xtb_root, cache_root):
    from src.dataset import build_dataset, filter_valid_indices
    from src.xtb_features import XTBFeatureStore
    key, identity = cache_identity(frame, settings, xtb_root)
    folder = Path(cache_root) / key
    receipt = folder / "complete.json"
    if receipt.exists():
        load_cache(folder, key, frame, load_graphs=False)
        print(f"Graph cache HIT: {len(frame)} records, {folder}", flush=True)
        return key, folder
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Graph cache MISS: build all {len(frame)} records once before fold training.", flush=True)
    unlabelled = frame.copy()
    unlabelled["G (kcal/mol)"] = 0.0
    valid = filter_valid_indices(unlabelled, default_catalyst="Co")
    if valid != list(range(len(frame))):
        raise ValueError("Invalid graph inputs in Co data; no rows may be silently omitted")
    store = XTBFeatureStore(xtb_root, cache_dir=folder / "xtb_parsed") if settings["use_xtb_features"] else None
    dataset, dimension = build_dataset(unlabelled, valid, torch.device("cpu"), default_catalyst="Co",
                                       xtb_store=store, **settings)
    if len(dataset) != len(frame) or [str(row[0]) for row in dataset] != frame.Name.astype(str).tolist():
        raise ValueError("Graph build lost/reordered samples")

    temporary = folder / "graphs.tmp.bin"
    dgl.save_graphs(str(temporary), [graph for row in dataset for graph in row[1:4]])
    temporary.replace(folder / "graphs.bin")
    dump(folder / "records.json", dict(names=frame.Name.astype(str).tolist(), temperatures=[float(row[4]) for row in dataset],
                                       global_dim=dimension, identity=identity, key=key,
                                       coverage=store.report() if store else {}))
    dump(receipt, dict(key=key, files={name: sha256(folder / name) for name in ("graphs.bin", "records.json")}))
    return key, folder


def load_cache(folder, key, frame, load_graphs=True):
    folder = Path(folder)
    receipt = read_json(folder / "complete.json")
    if receipt["key"] != key:
        raise ValueError("Graph cache identity mismatch")
    for name, digest in receipt["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Corrupt graph cache: {name}; do not reuse")
    record = read_json(folder / "records.json")
    if record["names"] != frame.Name.astype(str).tolist():
        raise ValueError("Graph cache sample identity/order mismatch")

    digest = hashlib.sha256(frame[INPUT_COLUMNS].to_json(orient="split", index=False, double_precision=15).encode()).hexdigest()
    if record["identity"]["inputs_sha256"] != digest:
        raise ValueError("Graph input values changed")
    if not load_graphs:
        return record
    graphs, _ = dgl.load_graphs(str(folder / "graphs.bin"))
    if len(graphs) != 3*len(frame):
        raise ValueError("Incomplete graph cache")
    targets = frame["G (kcal/mol)"].to_numpy(float)
    if not np.isfinite(targets).all():
        raise ValueError("Invalid targets")
    rows = [(name, *graphs[i*3:i*3+3], record["temperatures"][i], targets[i]) for i, name in enumerate(record["names"])]
    return rows, record


def select_samples(dataset, frame, hide_targets=False):
    by_name = {str(row[0]): row for row in dataset}
    if len(by_name) != len(dataset) or not frame.Name.is_unique:
        raise ValueError("Duplicate cached/sample identities")
    result = []
    for name in frame.Name.astype(str):
        row = by_name[name]
        result.append((*row[:5], 0.0 if hide_targets else row[5]))
    return result
