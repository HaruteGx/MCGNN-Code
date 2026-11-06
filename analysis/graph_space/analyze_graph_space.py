import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT, ROOT / "src/Co-transfer"):
    sys.path.insert(0, str(folder))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=["graph_summary", "model"], default="graph_summary")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--with-vif", action="store_true", help="Also evaluate local node/edge mean-feature collinearity")
    parser.add_argument("--smoke", action="store_true", help="24 rows for pipeline verification")
    args = parser.parse_args(argv)
    if min(args.threads, args.chunk_size, args.batch_size) < 1 or args.pca_components < 2 or args.perplexity <= 0 or args.iterations < 300 or not 0 <= args.seed < 2**32:
        parser.error("Require positive sizes/perplexity, PCA>=2, iterations>=300 and uint32 seed")
    output = (args.output_dir or ROOT / "outputs/model_representation" / (("smoke_" if args.smoke else "")+args.embedding)).resolve()
    if not output.is_relative_to(ROOT / "outputs/model_representation") or output == ROOT / "outputs/model_representation":
        parser.error("Use a run subdirectory of outputs/model_representation")
    if args.smoke != output.name.startswith("smoke_"):
        parser.error("Smoke output names must begin with smoke_; formal runs must not")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    import numpy as np
    import pandas as pd
    import torch
    import dgl
    import sklearn
    from rdkit import rdBase, RDLogger
    from threadpoolctl import threadpool_limits
    from src.ir_protocol import ir_run_paths, load_ir_test_predictions
    from src.data_splits import load_frozen_ir_split, checkpoint_split_config
    from src.dataset import build_dataset, filter_valid_indices
    from src.prediction import load_regression_checkpoint
    from src.xtb_features import XTBFeatureStore
    from transfer_to_co import feature_settings_from_checkpoint
    from analysis.graph_space.core import extract_vectors, project_space, nearest_neighbors, vif_table
    RDLogger.DisableLog("rdApp.warning")
    torch.set_num_threads(args.threads)
    paths = ir_run_paths(ROOT)
    development, test, split = load_frozen_ir_split(paths["development"], paths["test"], paths["manifest"], paths["full"])
    frame = pd.read_csv((paths["full"]), dtype={"Name": str})
    if not frame.Name.is_unique or frame.Name.isna().any() or not frame.Cat.eq("Ir").all():
        raise ValueError("Require uniquely identified Ir records only")
    if args.smoke:
        names = set(development.Name.astype(str).head(12)) | set(test.Name.astype(str).head(12))
        frame = frame.loc[frame.Name.isin(names)].reset_index(drop=True)
    records = frame[["Name", "G (kcal/mol)"]].copy()
    records["class_id"] = np.searchsorted([23., 25.], records["G (kcal/mol)"], side="right")
    records["split"] = np.where(records.Name.isin(set(test.Name.astype(str))), "test", "development")
    if not np.isfinite(records["G (kcal/mol)"]).all():
        raise ValueError("Nonfinite experimental labels")
    model, metadata, _ = load_regression_checkpoint(str(paths["stable_checkpoint"]), torch.device("cpu"))
    if (metadata.get("default_catalyst") != "Ir" or metadata.get("head") != "enthalpy_entropy_physics"
            or metadata.get("holdout_split") != checkpoint_split_config(split) or model.ablate_branches):
        raise ValueError("Requires the full current Ir H/S model matching the frozen split")
    model.eval()
    features = feature_settings_from_checkpoint(model.state_dict(), metadata)
    xtb_root = ROOT / "data" / "xtb_ir_features"
    store = XTBFeatureStore(xtb_root) if features["use_xtb_features"] else None
    xtb_sources = {}
    if store:
        xtb_sources["xtb_unique_molecule_features.csv"] = sha(xtb_root / "xtb_unique_molecule_features.csv")
        for _, row in store.jobs.values():
            for name in ("xtbout.json", "result.json", "xtbopt.xyz", "sp.out", "wbo", "vibspectrum"):
                path = xtb_root / "calculations" / row["role"] / row["job_id"] / name
                xtb_sources[path.relative_to(xtb_root).as_posix()] = sha(path) if path.exists() else "missing"
    input_columns = ["Name", "sol", "sub-type", "P", "L-type", "B-type", "Cat", "bpin_index_sub", "bpin_index Production", "T / C"]
    source_files = [ROOT / "src" / name for name in ("dataset.py", "graph_builders.py", "features.py", "models.py", "descriptors.py", "mcencoding.py", "xtb_features.py", "prediction.py")]
    source_files += [Path(__file__).with_name("core.py"), Path(__file__), ROOT / "src/Co-transfer/transfer_to_co.py"]
    signature = dict(version=1, input_sha256=hashlib.sha256(frame[input_columns].to_csv(index=False).encode()).hexdigest(),
                     checkpoint_sha256=sha(paths["stable_checkpoint"]), feature_settings=features,
                     xtb_sources=xtb_sources, chunk_size=args.chunk_size,
                     code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
                     versions=dict(torch=torch.__version__, rdkit=rdBase.rdkitVersion, dgl=dgl.__version__))
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "outputs/model_representation/vector_cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    dump(cache / "identity.json", signature)
    all_vectors, schema = [], None
    print(f"Ir graph-space analysis: {len(frame)} rows ({records.split.value_counts().to_dict()}), CPU; embedding={args.embedding}", flush=True)
    for start in range(0, len(frame), args.chunk_size):
        part = frame.iloc[start:start+args.chunk_size].copy().reset_index(drop=True)
        path = cache / f"chunk_{start:05d}.npz"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            evidence = json.loads(receipt.read_text(encoding="utf-8"))
            if sha(path) != evidence["sha256"]:
                raise ValueError(f"Changed vector cache: {path}")
            with np.load((path), allow_pickle=False) as stored:
                vectors = {k: stored[k] for k in stored.files}
            part_schema = evidence["schema"]
            status = "cache HIT"
        else:
            unlabelled = part.copy()
            unlabelled["G (kcal/mol)"] = 0.  
            valid = filter_valid_indices(unlabelled, default_catalyst="Ir")
            if valid != list(range(len(unlabelled))):
                raise ValueError("Invalid graph inputs; no rows may be silently omitted")
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                dataset, _ = build_dataset(unlabelled, valid, torch.device("cpu"), default_catalyst="Ir", xtb_store=store, **features)
            ((cache / f"chunk_{start:05d}_build.log")).write_text(log.getvalue(), encoding="utf-8")
            if [str(s[0]) for s in dataset] != part.Name.tolist():
                raise ValueError("Graph build dropped or reordered records")
            vectors, part_schema = extract_vectors(model, dataset, args.batch_size)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed((temporary), **vectors)
            temporary.replace(path)
            dump(receipt, dict(sha256=sha(path), schema=part_schema))
            del dataset
            status = "graphs built + vectors cached"
        if vectors["names"].tolist() != part.Name.tolist() or any(not np.isfinite(v).all() for k, v in vectors.items() if k != "names"):
            raise ValueError("Vector sample alignment or finite-value check failed")
        if schema is not None and schema != part_schema:
            raise ValueError("Feature schema changed across cache chunks")
        schema = part_schema
        all_vectors.append(vectors)
        print(f"{min(start+len(part), len(frame))}/{len(frame)}: {status}", flush=True)
    vectors = {name: np.concatenate([v[name] for v in all_vectors]) for name in all_vectors[0]}

    saved = load_ir_test_predictions(ROOT, test)
    mask = records.split.eq("test").to_numpy()
    error = np.abs(vectors["predicted_g"][mask]-saved.loc[records.Name[mask], "Predicted G (kcal/mol)"].to_numpy()).max()
    if error > 2e-4:
        raise ValueError(f"Graphs fail to reproduce saved test predictions: {error}")
    projection_settings = dict(vector_cache_key=key, representation=args.embedding, seed=args.seed,
                               perplexity=args.perplexity, pca=args.pca_components, iterations=args.iterations,
                               sklearn=sklearn.__version__, numpy=np.__version__)
    projection_file, projection_meta = output / "projection.npz", output / "projection.json"
    reuse = False
    if projection_meta.exists() and projection_file.exists():
        previous = json.loads(projection_meta.read_text(encoding="utf-8"))
        reuse = previous["settings"] == projection_settings and previous["sha256"] == sha(projection_file)
    with threadpool_limits(limits=args.threads):
        if reuse:
            with np.load((projection_file), allow_pickle=False) as data:
                coords, reduced = data["coordinates"], data["pca_vectors"]
            projection_audit = previous["audit"]
            print("t-SNE cache HIT", flush=True)
        else:
            print("Standardization, PCA and t-SNE...", flush=True)
            coords, reduced, projection_audit = project_space(vectors[args.embedding], args.seed, args.perplexity, args.iterations, args.pca_components)
            np.savez_compressed((projection_file), coordinates=coords, pca_vectors=reduced)
            dump(projection_meta, dict(settings=projection_settings, audit=projection_audit, sha256=sha(projection_file)))
        index, distance, cosine = nearest_neighbors(reduced)
        target = records["G (kcal/mol)"].to_numpy()
        neighbors = pd.DataFrame(dict(Name=records.Name, nearest_Name=records.Name.iloc[index].to_numpy(), distance=distance,
                                      cosine_similarity=cosine, signed_delta_g=target-target[index], absolute_delta_g=np.abs(target-target[index])))
        vif = vif_table(vectors["local"], schema["local"]) if args.with_vif else None
    coordinates_table = records.copy()
    coordinates_table[["tsne_1", "tsne_2"]] = coords
    coordinates_table.to_csv((output / "tsne_coordinates.csv"), index=False, encoding="utf-8-sig")
    neighbors.to_csv((output / "nearest_neighbors.csv"), index=False, encoding="utf-8-sig")
    if vif is not None:
        vif.to_csv((output / "local_feature_vif.csv"), index=False, encoding="utf-8-sig")
    dump(output / "feature_schema.json", schema)
    audit = dict(n=len(records), full_ir_n=len(development)+len(test), split_counts=records.split.value_counts().to_dict(),
                 smoke=args.smoke, model_selected_fold=metadata.get("cv_fold"), representation=args.embedding,
                 graph_summary_dimensions=vectors["graph_summary"].shape[1], model_latent_dimensions=vectors["model"].shape[1],
                 node_feature_dim=len(__import__("src.features", fromlist=["NODE_FEATURE_NAMES"]).NODE_FEATURE_NAMES),
                 saved_test_prediction_max_abs_error=float(error), classification_boundaries=[23, 25],
                 source_hashes={k: sha(paths[k]) for k in ("full", "development", "test", "manifest", "stable_checkpoint")},
                 vector_cache_key=key, projection=projection_audit, xtb_coverage=store.report() if store else {},
                 notes=["Graph-space coordinates and nearest-neighbor statistics."])
    dump(output / "analysis_audit.json", audit)
    print(f"Completed {'SMOKE' if args.smoke else 'full Ir'} graph-space analysis: {output}", flush=True)


if __name__ == "__main__":
    main()
