from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT, ROOT / "src/Co-transfer"):
    sys.path.insert(0, str(folder))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "models/Ir checkpoints.pth")
    p.add_argument("--xtb-root", type=Path, default=ROOT / "data" / "xtb_ir_features")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--random-seed", type=int, default=20260918)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--groups-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="First 12 test samples and 3 repeats")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    if args.repeats < 2 or min(args.batch_size, args.threads) < 1 or not 0 <= args.random_seed < 2**32:
        p.error("Require >=2 repeats, positive batch/threads and uint32 seed")
    if args.smoke:
        args.repeats = 3
    allowed = ROOT / "outputs/interpretability/feature_importance"
    args.output_dir = (args.output_dir or allowed / ("smoke_frozen_test_v1" if args.smoke else "reproduced")).resolve()
    if not args.output_dir.is_relative_to(allowed.resolve()) or args.output_dir == allowed.resolve():
        p.error("Use a run subdirectory of outputs/interpretability/feature_importance")
    if args.smoke != args.output_dir.name.startswith("smoke_"):
        p.error("Smoke directory must begin with smoke_; formal directory must not")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / ".matplotlib")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    import numpy as np
    import pandas as pd
    import torch
    from co_transfer_holdout.protocol import dump, read_json, sha256
    from src.data_splits import load_frozen_ir_split, checkpoint_split_config
    from src.ir_protocol import ir_run_paths, load_ir_test_predictions
    from src.dataset import build_dataset, filter_valid_indices
    from src.prediction import load_regression_checkpoint
    from src.xtb_features import get_xtb_store
    from transfer_to_co import feature_settings_from_checkpoint
    from analysis.feature_importance.schema import feature_groups
    from analysis.feature_importance.permutation import FrozenPredictor, evaluate_group, metrics, verify_intervention
    from analysis.feature_importance.summary import summarize

    torch.set_num_threads(args.threads)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    paths = ir_run_paths(ROOT)

    _, test, manifest = load_frozen_ir_split(paths["development"], paths["test"], paths["manifest"], paths["full"])
    model, metadata, _ = load_regression_checkpoint(str(args.checkpoint), device)
    if metadata.get("default_catalyst") != "Ir" or metadata.get("holdout_split") != checkpoint_split_config(manifest):
        raise ValueError("Checkpoint must be an Ir model trained on this exact frozen development split")
    if metadata.get("head") != "enthalpy_entropy_physics" or model.ablate_branches:
        raise ValueError("Requires the full H/S thermodynamic model")
    feature_settings = feature_settings_from_checkpoint(model.state_dict(), metadata)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(checkpoint_sha256=sha256(args.checkpoint), split_sha256=sha256(paths["manifest"]),
                    xtb_manifest_sha256=sha256(args.xtb_root / "xtb_unique_molecule_features.csv"),
                    code_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sorted([
                        *Path(__file__).parent.glob("*.py"), *(ROOT / "src").glob("*.py"),
                        ROOT / "src/Co-transfer/transfer_to_co.py"])})
    output = args.output_dir
    settings_path = output / "settings.json"
    if settings_path.exists() and not (read_json(settings_path) == settings):
        raise ValueError("Importance settings/code changed; choose a new --output-dir to preserve existing results")
    output.mkdir(parents=True, exist_ok=True)
    if not settings_path.exists():
        dump(settings_path, settings)
    if ((output / "complete.json")).exists():
        for name, digest in read_json(output / "complete.json")["files"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"Completed importance output changed: {name}")
        print(f"Existing completed importance analysis verified: {output}", flush=True)
        return
    if args.smoke:
        test = test.iloc[:12].copy()
    truth = test["G (kcal/mol)"].to_numpy(float)
    if not np.isfinite(truth).all():
        raise ValueError("Invalid evaluation targets")
    store = get_xtb_store(str(args.xtb_root)) if feature_settings["use_xtb_features"] else None
    print(f"Ir frozen-model importance: {len(test)} test records, {args.repeats} permutations/group, device={device}", flush=True)
    dataset, _ = build_dataset(test, filter_valid_indices(test, default_catalyst="Ir"), torch.device("cpu"),
                               default_catalyst="Ir", xtb_store=store, **feature_settings)
    if len(dataset) != len(test) or [str(sample[0]) for sample in dataset] != test.Name.astype(str).tolist():
        raise ValueError("Every frozen test record must produce a graph; no samples may be dropped")
    predictor = FrozenPredictor(model, dataset, device, args.batch_size)
    groups = feature_groups(model, metadata, not args.groups_only)
    dump(output / "feature_groups.json", groups)
    dump(output / "xtb_coverage.json", store.report() if store else {})

    examples = [next(g for g in groups if g["kind"] == kind) for kind in ("branch", "local", "global", "temperature")]
    examples.extend([next(g for g in groups if g["name"] == "reactant graph"),
                     next(g for g in groups if g["name"] == "reactant local graph features"),
                     next(g for g in groups if g["name"] == "Ligand 2D descriptors")])
    if not args.groups_only:
        examples.append(next(g for g in groups if g["strata"]))
    errors = {g["name"]: verify_intervention(predictor, dataset, g, predictor.permutation(g, args.random_seed)) for g in examples}
    baseline = predictor.baseline.cpu().numpy()
    baseline_scores = metrics(truth, baseline[:, 0])
    audit = dict(baseline_max_direct_difference=predictor.max_direct_error, intervention_max_differences=errors,
                 n=len(test), selected_cv_fold=metadata.get("cv_fold"), baseline_metrics=baseline_scores,
                 smoke=args.smoke, method="Repeated grouped permutation importance on frozen Ir record holdout",
                 classification_boundaries=[23., 25.], uncertainty="Permutation variability",
                 excluded_from_scope="Grouped graph and descriptor permutation")
    _, full_test, _ = load_frozen_ir_split(paths["development"], paths["test"], paths["manifest"], paths["full"])
    saved = load_ir_test_predictions(ROOT, full_test).loc[test.Name.astype(str)]
    error = np.max(np.abs(saved["Predicted G (kcal/mol)"].to_numpy()-baseline[:, 0]))
    if error > 2e-4:
        raise ValueError(f"Predictions do not reproduce the saved Ir run: max difference={error}")
    audit["saved_prediction_max_difference"] = float(error)
    dump(output / "analysis_audit.json", audit)
    pd.DataFrame({"Name": test.Name, "True G (kcal/mol)": truth, "Predicted G (kcal/mol)": baseline[:, 0],
                  "Predicted Delta H (kcal/mol)": baseline[:, 1], "Predicted Delta S (cal/mol/K)": baseline[:, 2],
                  "True Class": np.searchsorted([23., 25.], truth, side="right"),
                  "Predicted Class": np.searchsorted([23., 25.], baseline[:, 0], side="right")}).to_csv((output / "baseline_predictions.csv"), index=False)
    results = []
    for index, group in enumerate(groups):
        folder = output / "groups"
        folder.mkdir(exist_ok=True)
        table, receipt = folder / f"group_{index:03d}.csv", folder / f"group_{index:03d}.json"
        if receipt.exists():
            evidence = read_json(receipt)
            if evidence["group"] != group or evidence["sha256"] != sha256(table):
                raise ValueError("Saved group result changed")
            rows = pd.read_csv(table)
        else:
            rows = pd.DataFrame(evaluate_group(predictor, group, truth, args.repeats, args.random_seed))
            rows.to_csv((table), index=False)
            dump(receipt, {"group": group, "sha256": sha256(table)})
        results.append(rows)
        if (index+1) % 10 == 0 or index+1 == len(groups):
            print(f"Importance groups complete: {index+1}/{len(groups)}", flush=True)
    raw = pd.concat(results, ignore_index=True)
    summary = summarize(raw)
    raw.to_csv((output / "permutation_repeats.csv"), index=False)
    summary.to_csv((output / "importance_summary.csv"), index=False)
    report = ["# Ir 特征重要性", "",
              f"样本数：{len(test)}；每组置换次数：{args.repeats}。",
              f"基线 ACC={baseline_scores['accuracy']:.4%}，R²={baseline_scores['r2']:.4f}，MAE={baseline_scores['mae']:.4f} kcal/mol。",
              "结果见 importance_summary.csv、permutation_repeats.csv 和 feature_groups.json。"]
    ((output / "report.md")).write_text("\n".join(report)+"\n", encoding="utf-8")
    artifacts = ["settings.json", "baseline_predictions.csv", "analysis_audit.json", "feature_groups.json", "xtb_coverage.json",
                 "permutation_repeats.csv", "importance_summary.csv", "report.md"]
    dump(output / "complete.json", {"smoke": args.smoke, "files": {name: sha256(output / name) for name in artifacts}})
    print(f"Completed {'SMOKE' if args.smoke else 'formal'} Ir importance analysis: {output}", flush=True)


if __name__ == "__main__":
    main()
