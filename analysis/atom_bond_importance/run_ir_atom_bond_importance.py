import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT, ROOT / "src/Co-transfer"):
    sys.path.insert(0, str(path))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--samples", nargs="+", help="Exact Ir test Name values")
    selection.add_argument("--all-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/interpretability/ir_atom_bond_importance/complex_acs_v3")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.steps < 2 or args.max_steps < args.steps*2 or args.threads < 1:
        parser.error("Require steps >= 2, max-steps >= 2*steps and threads >= 1")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs/interpretability/ir_atom_bond_importance") or output == ROOT / "outputs/interpretability/ir_atom_bond_importance":
        parser.error("Use a run subdirectory of outputs/interpretability/ir_atom_bond_importance")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    import dgl
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors
    from src.ir_protocol import ir_run_paths, load_ir_test_predictions
    from src.data_splits import load_frozen_ir_split, checkpoint_split_config
    from src.prediction import load_regression_checkpoint
    from src.dataset import build_dataset, filter_valid_indices, parse_reaction_site_indices
    from src.xtb_features import XTBFeatureStore
    from src.features import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES
    from transfer_to_co import feature_settings_from_checkpoint
    from analysis.atom_bond_importance.attribution import explain
    torch.set_num_threads(args.threads)
    paths = ir_run_paths(ROOT)
    _, test, manifest = load_frozen_ir_split(paths["development"], paths["test"], paths["manifest"], paths["full"])
    test = test.copy()
    test["Name"] = test.Name.astype(str)
    if args.samples:
        if len(set(args.samples)) != len(args.samples) or set(args.samples)-set(test.Name):
            raise ValueError("Sample names must be unique and belong to the frozen Ir test set")
        selected = test.set_index("Name").loc[args.samples].reset_index()
        rule = "User-specified frozen test sample names"
    elif args.all_test:
        selected, rule = test, "All frozen Ir test samples"
    else:

        test["_class"] = np.searchsorted([23, 25], test["G (kcal/mol)"], side="right")
        test["_atoms"] = test["sub-type"].map(lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms())
        test["_complexity"] = test["sub-type"].map(lambda s: Descriptors.BertzCT(Chem.MolFromSmiles(s)))
        test["_canonical"] = test["sub-type"].map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
        chosen, used = [], set()
        for category in (0, 1, 2):
            candidates = test.loc[test._class.eq(category) & ~test._canonical.isin(used)].sort_values(
                ["_complexity", "_atoms", "Name"], ascending=[False, False, True])
            if candidates.empty:
                raise ValueError("Cannot select distinct complex substrates covering three classes")
            chosen.append(candidates.iloc[[0]])
            used.add(candidates.iloc[0]._canonical)
        selected = pd.concat(chosen).drop(columns=["_class", "_atoms", "_complexity", "_canonical"])
        rule = "Highest RDKit BertzCT in each true class, distinct substrates; heavy atoms then Name break ties"
    selected = selected.reset_index(drop=True)
    checkpoint = paths["stable_checkpoint"]
    model, metadata, _ = load_regression_checkpoint(str(checkpoint), torch.device("cpu"))
    if (metadata.get("default_catalyst") != "Ir" or metadata.get("head") != "enthalpy_entropy_physics"
            or metadata.get("holdout_split") != checkpoint_split_config(manifest) or model.ablate_branches):
        raise ValueError("Requires the full H/S Ir checkpoint for this exact frozen split")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    features = feature_settings_from_checkpoint(model.state_dict(), metadata)
    xtb = ROOT / "data" / "xtb_ir_features"

    xtb_hashes = {"xtb_unique_molecule_features.csv": sha(xtb / "xtb_unique_molecule_features.csv")}
    source_names = {"xtbout.json", "result.json", "xtbopt.xyz", "sp.out", "wbo", "vibspectrum"}
    for path in sorted((xtb / "calculations").rglob("*")):
        if path.is_file() and path.name in source_names:
            xtb_hashes[path.relative_to(xtb).as_posix()] = sha(path)
    settings = dict(version=3, samples=selected.Name.tolist(), selection_rule=rule,
                    checkpoint_sha256=sha(checkpoint), split_sha256=sha(paths["manifest"]),
                    selected_inputs_sha256=hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest(),
                    feature_settings=features, steps=args.steps, max_steps=args.max_steps,
                    threads=args.threads, device="cpu", xtb_sources=xtb_hashes,
                    code_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted([
                        *Path(__file__).parent.glob("*.py"), *(ROOT / "src").glob("*.py"),
                        ROOT / "src/Co-transfer/transfer_to_co.py"])},
                    versions=dict(torch=torch.__version__, dgl=dgl.__version__, rdkit=rdBase.rdkitVersion))
    settings_file = output / "settings.json"
    if settings_file.exists() and json.loads(settings_file.read_text(encoding="utf-8")) != settings:
        raise ValueError("Settings or source changed; choose a new --output-dir to preserve existing results")
    dump(settings_file, settings)
    if ((output / "complete.json")).exists():
        receipt = json.loads(((output / "complete.json")).read_text(encoding="utf-8"))
        for name, digest in receipt["files"].items():
            if sha(output / name) != digest:
                raise ValueError(f"Completed output changed: {name}")
        print(f"Completed result verified: {output}")
        return
    print(f"Ir atom/bond IG: {selected.Name.tolist()}; CPU", flush=True)
    cache = output / "graph_cache"
    cache.mkdir(exist_ok=True)
    if ((cache / "complete.json")).exists():
        receipt = json.loads(((cache / "complete.json")).read_text())
        for name, digest in receipt.items():
            if sha(cache / name) != digest:
                raise ValueError(f"Graph cache changed: {name}")
        graphs, _ = dgl.load_graphs(str(cache / "graphs.bin"))
        temperatures = json.loads(((cache / "temperatures.json")).read_text())
        if len(graphs) != len(selected)*3 or len(temperatures) != len(selected):
            raise ValueError("Incomplete graph cache")
        dataset = [(row.Name, *graphs[3*i:3*i+3], temperatures[i], row["G (kcal/mol)"])
                   for i, row in selected.iterrows()]
        print("Graph cache HIT", flush=True)
    else:
        unlabelled = selected.copy()
        unlabelled["G (kcal/mol)"] = 0.
        valid = filter_valid_indices(unlabelled, default_catalyst="Ir")
        if valid != list(range(len(selected))):
            raise ValueError("Selected rows must all produce graphs")
        dataset, _ = build_dataset(unlabelled, valid, torch.device("cpu"), default_catalyst="Ir",
                                  xtb_store=XTBFeatureStore(xtb, cache_dir=cache / "xtb_parsed"), **features)
        if [str(s[0]) for s in dataset] != selected.Name.tolist():
            raise ValueError("Graph build lost or reordered samples")
        dgl.save_graphs(str(cache / "graphs.bin"), [g for s in dataset for g in s[1:4]])
        dump(cache / "temperatures.json", [s[4] for s in dataset])
        dump(cache / "complete.json", {name: sha(cache / name) for name in ("graphs.bin", "temperatures.json")})
    saved_predictions = load_ir_test_predictions(ROOT, test)
    results, summary, artifacts = [], [], ["settings.json"]
    for i, (sample, (_, row)) in enumerate(zip(dataset, selected.iterrows())):
        folder = f"sample_{i+1:03d}"
        sample_out = output / folder
        sample_out.mkdir(exist_ok=True)
        sample_receipt = sample_out / "complete.json"
        if sample_receipt.exists():
            for name, digest in json.loads(sample_receipt.read_text()).items():
                if sha(sample_out / name) != digest:
                    raise ValueError(f"Saved sample result changed: {sample[0]}/{name}")
            audit = json.loads(((sample_out / "audit.json")).read_text())
            atom_table, bond_table = pd.read_csv((sample_out / "atoms.csv")), pd.read_csv((sample_out / "bonds.csv"))
            atom_values, bond_values = atom_table.attribution_kcal_mol.to_numpy(), bond_table.attribution_kcal_mol.to_numpy()
            mol = Chem.MolFromSmiles(row["sub-type"])
        else:
            print(f"Explain {i+1}/{len(dataset)}: {sample[0]}", flush=True)
            mol, pairs, atom_values, bond_values, node, edge, audit = explain(model, sample, row["sub-type"], args.steps, args.max_steps)
            audit.update(sample_name=str(row.Name), substrate_smiles=row["sub-type"],
                         true_class=int(np.searchsorted([23, 25], row["G (kcal/mol)"], side="right")),
                         target="delta_g after physical H/S head", method="joint atom/bond feature Integrated Gradients",
                         reference="zero substrate local feature values, fixed role/missingness flags",
                         held_fixed="topology, all global features, product, solvent, catalyst, boron, ligand, temperature, reaction-site flags",
                         scope="Substrate local-feature attribution")
            error = abs(audit["predicted_delta_g"]-saved_predictions.loc[row.Name, "Predicted G (kcal/mol)"])
            audit["saved_prediction_abs_error"] = float(error)
            dump(sample_out / "audit.json", audit)
            if error > 2e-4:
                raise ValueError("Original forward does not reproduce saved Ir predictions")
            if not audit["converged"]:
                raise ValueError(f"IG not numerically converged for {row.Name}; inspect audit and increase --max-steps in a new output directory")
            atom_table = pd.DataFrame(dict(atom_index_1based=np.arange(mol.GetNumAtoms())+1,
                                          element=[a.GetSymbol() for a in mol.GetAtoms()], attribution_kcal_mol=atom_values))
            bond_table = pd.DataFrame([dict(bond_index_1based=b.GetIdx()+1, atom1_1based=b.GetBeginAtomIdx()+1,
                                           atom2_1based=b.GetEndAtomIdx()+1, bond_type=str(b.GetBondType()),
                                           directed_edge_ids=str(pairs[b.GetIdx()]), attribution_kcal_mol=float(bond_values[b.GetIdx()]))
                                       for b in mol.GetBonds()], columns=["bond_index_1based", "atom1_1based", "atom2_1based", "bond_type", "directed_edge_ids", "attribution_kcal_mol"])
            atom_table.to_csv((sample_out / "atoms.csv"), index=False)
            bond_table.to_csv((sample_out / "bonds.csv"), index=False)
            pd.DataFrame(node[:mol.GetNumAtoms()], columns=NODE_FEATURE_NAMES).to_csv((sample_out / "atom_channels.csv"), index=False)
            pd.DataFrame([edge[pair].sum(axis=0) for pair in pairs], columns=EDGE_FEATURE_NAMES).to_csv((sample_out / "bond_channels.csv"), index=False)
            dump(sample_receipt, {name: sha(sample_out / name) for name in ("audit.json", "atoms.csv", "bonds.csv", "atom_channels.csv", "bond_channels.csv")})
        sites = parse_reaction_site_indices(row["bpin_index_sub"], "bpin_index_sub", i)
        results.append(dict(name=str(row.Name), folder=folder, mol=mol, atoms=atom_values, bonds=bond_values, sites=sites, audit=audit))
        summary.append(dict(Name=row.Name, predicted_delta_g=audit["predicted_delta_g"], predicted_class=audit["predicted_class"],
                            true_class=audit["true_class"], reference_delta_g=audit["reference_delta_g"],
                            attribution_sum=audit["attribution_sum"], converged=audit["converged"],
                            max_integration_steps=audit["integration_history"][-1]["steps"]))
        print(f"{row.Name}: G={audit['predicted_delta_g']:.4f}, class={audit['predicted_class']}, IG converged", flush=True)
    pd.DataFrame(summary).to_csv((output / "summary.csv"), index=False)
    artifacts += [p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
                  and p.suffix in {".csv", ".png", ".svg", ".pdf", ".json"}
                  and "graph_cache" not in p.parts and ".matplotlib" not in p.parts and p.name != "settings.json"]
    dump(output / "complete.json", {"files": {name: sha(output / name) for name in artifacts}})
    print(f"Saved atom/bond CSV values and numerical audits: {output}", flush=True)


if __name__ == "__main__":
    main()
