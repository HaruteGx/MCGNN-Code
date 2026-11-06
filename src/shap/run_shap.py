from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/interpretability/shap"
for folder in (ROOT, ROOT / "src/Co-transfer"):
    sys.path.insert(0, str(folder))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["groups", "descriptors"], default="descriptors")
    parser.add_argument("--fingerprints", choices=["include", "fixed"], default="include",
                        help="include: original mode; fixed: keep target fingerprints but attribute only other features")
    parser.add_argument("--background", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=16, help="Independent forward/reverse order pairs")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    from grouped_shapley import disjoint_groups, attribution_partition, CoalitionGame, estimate_shapley
    if args.background < 1 or args.repeats < 2 or not 0 <= args.seed < 2**32:
        parser.error("Require background>=1, repeats>=2 and uint32 seed")
    if args.smoke:
        args.background, args.repeats = 3, 2
    if args.level == "descriptors":
        run_name = "non_fingerprint_main" if args.fingerprints == "fixed" else "including_fingerprints_supplementary"
    else:
        run_name = "groups_fixed_temperature" + ("_fixed_fingerprints" if args.fingerprints == "fixed" else "")
    if args.smoke:
        run_name = "smoke_" + run_name
    output = args.output_dir or OUTPUT_ROOT / run_name
    output = output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        parser.error("Outputs must be a run subdirectory under outputs/interpretability/shap")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; select a new --output-dir to preserve saved results")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import RDLogger
    from src.data_splits import load_frozen_ir_split, checkpoint_split_config
    from src.ir_protocol import ir_run_paths
    from src.prediction import load_regression_checkpoint
    from src.dataset import build_dataset, filter_valid_indices
    from src.xtb_features import get_xtb_store
    from transfer_to_co import feature_settings_from_checkpoint
    from analysis.feature_importance.permutation import FrozenPredictor, metrics
    RDLogger.DisableLog("rdApp.warning")
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    paths = ir_run_paths(ROOT)
    dev, test, manifest = load_frozen_ir_split(paths["development"], paths["test"], paths["manifest"], paths["full"])
    model, metadata, _ = load_regression_checkpoint(str(paths["stable_checkpoint"]), device)
    if (metadata.get("default_catalyst") != "Ir" or metadata.get("head") != "enthalpy_entropy_physics"
            or metadata.get("holdout_split") != checkpoint_split_config(manifest) or model.ablate_branches):
        raise ValueError("Requires current full Ir checkpoint and matching frozen split")
    model.eval()
    full_groups = disjoint_groups(model, metadata, args.level)
    groups, fixed_groups = attribution_partition(full_groups, args.fingerprints)
    rng = np.random.default_rng(args.seed)
    if args.background > len(dev):
        raise ValueError("Requested background exceeds development pool")

    background = dev.iloc[np.sort(rng.choice(len(dev), args.background, replace=False))].copy()
    if args.smoke:
        test = test.iloc[:3].copy()
    if set(background.Name) & set(test.Name):
        raise ValueError("Background must not include test records")
    combined = pd.concat([background, test], ignore_index=True)
    settings = dict(level=args.level, seed=args.seed, repeats=args.repeats, background=args.background,
                    fingerprints=args.fingerprints,
                    fingerprint_policy="Target fingerprints fixed in every coalition" if fixed_groups else "Fingerprints included as whole players",
                    smoke=args.smoke, device=str(device), method="Grouped interventional Monte Carlo Shapley; forward/reverse orders",
                    output="Predicted Delta G (kcal/mol)",
                    temperature_policy="Fixed at each target record's experimental T in all coalitions; excluded from SHAP players",
                    background_names=background.Name.tolist(), test_names=test.Name.tolist(),
                    sources={str(p.relative_to(ROOT)): sha(p) for p in [paths["stable_checkpoint"], paths["development"], paths["test"], paths["manifest"]]},
                    code_hashes={str(p.relative_to(ROOT)): sha(p) for p in [*Path(__file__).parent.glob("*.py"),
                        * (ROOT / "src").glob("*.py"), * (ROOT / "analysis/feature_importance").glob("*.py")]})
    dump(output / "settings.json", settings)
    dump(output / "feature_groups.json", groups)
    dump(output / "fixed_feature_groups.json", fixed_groups)
    background.to_csv(output / "background_records.csv", index=False)
    features = feature_settings_from_checkpoint(model.state_dict(), metadata)
    store = get_xtb_store(str(ROOT / "data" / "xtb_ir_features")) if features["use_xtb_features"] else None
    print(f"Building {len(combined)} records; {len(groups)} nonoverlapping SHAP players; {device}", flush=True)
    dataset, _ = build_dataset(combined, filter_valid_indices(combined, default_catalyst="Ir"), torch.device("cpu"),
                              default_catalyst="Ir", xtb_store=store, **features)
    if [str(s[0]) for s in dataset] != combined.Name.astype(str).tolist():
        raise ValueError("Graph construction dropped or reordered records")
    predictor = FrozenPredictor(model, dataset, device)
    dump(output / "xtb_coverage.json", store.report() if store else {})

    np.savez_compressed(output / "cached_model_factors.npz", **{k:v.cpu().numpy() for k,v in predictor.data.items()})
    rows, audits, repeated = [], [], []
    bg_indices = list(range(len(background)))
    for i, (_, row) in enumerate(test.iterrows()):
        game = CoalitionGame(predictor, groups, i+len(background), bg_indices, fixed_groups=fixed_groups)
        verification = game.verify(dataset, seed=args.seed+i) if i < 3 else None
        result = estimate_shapley(game, len(groups), args.repeats, args.seed+i)
        direct = float(predictor.data["direct"][i+len(background), 0])
        target_t = predictor.data["temperature"][i+len(background), 0]
        expected = game.reference_value()
        if max(abs(result["residual"]), abs(result["prediction"]-direct), abs(result["base"]-expected)) > 2e-4:
            raise ValueError("Additivity or endpoint reconstruction failed")
        repeated.append(result["repeats"])
        half = args.repeats//2
        half_phi = result["repeats"][:half].mean(axis=0)
        for j, g in enumerate(groups):
            rows.append(dict(Name=row.Name, feature=g["name"], kind=g["kind"], shap_value=result["values"][j],
                             mc_sem=result["sem"][j], half_budget_shap=half_phi[j]))
        audits.append(dict(Name=row.Name, true_g=row["G (kcal/mol)"], predicted_g=direct, base_value=result["base"],
                           fixed_temperature_kelvin=float(target_t), shap_sum=float(result["values"].sum()), additivity_error=result["residual"],
                           hybrid_forward_max_error=verification))

        pd.DataFrame(rows).to_csv(output / "shap_values.csv", index=False)
        pd.DataFrame(audits).to_csv(output / "sample_audit.csv", index=False)
        if (i+1)%10 == 0 or i+1 == len(test):
            print(f"SHAP samples {i+1}/{len(test)}", flush=True)
    values = pd.DataFrame(rows)
    np.savez_compressed(output / "order_pair_estimates.npz", estimates=np.stack(repeated),
                        sample_names=test.Name.astype(str).to_numpy(), feature_names=np.asarray([g["name"] for g in groups]))
    summary = values.groupby(["feature", "kind"], sort=False).agg(mean_abs_shap=("shap_value", lambda s:s.abs().mean()),
                  mean_signed_shap=("shap_value", "mean"), mean_mc_sem=("mc_sem", "mean"),
                  half_budget_mean_abs=("half_budget_shap", lambda s:s.abs().mean())).reset_index()
    summary = summary.sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(output / "feature_importance.csv", index=False)
    stability = summary.mean_abs_shap.corr(summary.half_budget_mean_abs, method="spearman")
    scores = metrics(test["G (kcal/mol)"], [a["predicted_g"] for a in audits])
    dump(output / "analysis_audit.json", dict(n=len(test), background_n=len(background), feature_count=len(groups),
         baseline_metrics=scores, max_direct_cache_error=predictor.max_direct_error,
         fingerprint_policy=settings["fingerprint_policy"], fixed_fingerprint_groups=len(fixed_groups),
         max_additivity_error=max(abs(a["additivity_error"]) for a in audits),
         half_full_rank_spearman=float(stability), interpretation="Monte Carlo Shapley estimates",
         background_policy="Development-record background; local-graph players; fixed target temperature"))
    report = ["# Ir grouped SHAP", "",
              f"Test records: {len(test)}; background records: {len(background)}; players: {len(groups)}.",
              f"Order pairs: {args.repeats}; half/full rank Spearman: {stability:.4f}.",
              "SHAP values are expressed in kcal/mol.",
              "Results: shap_values.csv, feature_importance.csv and sample_audit.csv.",
              "", "Top features:", ""]
    report += [f"- {r.feature}: {r.mean_abs_shap:.4f} kcal/mol" for r in summary.head(15).itertuples()]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    dump(output / "complete.json", {"files":{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name != "complete.json"}})
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
