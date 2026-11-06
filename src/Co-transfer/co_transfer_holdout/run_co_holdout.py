from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/Co-transfer"))
sys.path.insert(0, str(ROOT))

from co_transfer_holdout.protocol import (
    IR_CHECKPOINT, TARGET, aligned_predictions, combine_reports, dump, isolated_output,
    prepare, read_json, sha256, snapshot_current, verify_split, write_results,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/Database Co.csv")
    parser.add_argument("--checkpoint", type=Path, default=IR_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-seed", type=int, choices=[126], default=126)
    parser.add_argument("--cv-seed", type=int, choices=[10], default=10)
    parser.add_argument("--training-seed", type=int, default=666)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=150)
    parser.add_argument("--co-only-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--xtb-root", type=Path, default=ROOT / "data" / "xtb_Co_features")
    parser.add_argument("--components", choices=("all", "neural", "baselines"), default="all")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="Skip new evaluation plots; original training plot behaviour is retained.")
    parser.add_argument("--smoke", action="store_true", help="One epoch per neural stage and smaller RFs; always a separate smoke run.")
    args = parser.parse_args(argv)
    if min(args.stage1_epochs, args.stage2_epochs, args.co_only_epochs, args.batch_size, args.native_threads) < 1:
        parser.error("Epoch, batch and thread settings must be positive.")
    tag = f"{'smoke_' if args.smoke else ''}co_old_seed_{args.split_seed}_test_14"
    args.output_dir = isolated_output(args.output_dir or ROOT / "outputs/baselines/co_transfer_holdout" / tag)
    if args.smoke and not args.output_dir.name.startswith("smoke_"):
        parser.error("Smoke experiments must use an output directory whose name starts with smoke_.")
    if args.smoke:
        args.stage1_epochs = args.stage2_epochs = args.co_only_epochs = 1
    return args


def validate_ir_source(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("default_catalyst") != "Ir" or metadata.get("head") != "enthalpy_entropy_physics":
        raise ValueError("Start from the Ir thermodynamic checkpoint, never a Co-trained checkpoint.")
    if metadata.get("model_config", {}).get("ablate_branches") or metadata.get("graph_ablation", {}).get("disabled_branches"):
        raise ValueError("The source checkpoint must retain all three graph branches.")
    if checkpoint["model_state_dict"]["fc2.weight"].shape[0] != 2:
        raise ValueError("Expected the two-output Delta-H / Delta-S head.")
    return {"sha256": sha256(path), "task": metadata.get("task"), "head": metadata["head"],
            "default_catalyst": metadata["default_catalyst"]}


def freeze_source(args):
    source = args.checkpoint.resolve()
    provenance = validate_ir_source(source)
    destination = args.output_dir / "inputs/ir_source_checkpoint.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != provenance["sha256"]:
            raise ValueError("Frozen Ir source differs from the requested checkpoint.")
    else:
        shutil.copy2(source, destination)
    dump(destination.with_suffix(".json"), {**provenance, "original_source": str(source)})
    return destination, provenance


def training_commands(args, source):
    root = args.output_dir
    common = ["--data", str(root / "split/development.csv"), "--xtb_root", str(args.xtb_root.resolve()),
              "--cv_seed", str(args.cv_seed), "--training_seed", str(args.training_seed),
              "--cv_strategy", "group_source_target", "--batch_size", str(args.batch_size)]
    transfer = [sys.executable, str(ROOT / "src/Co-transfer/transfer_to_co.py"), "--regression-baseline", *common,
                "--checkpoint", str(source), "--output_dir", str(root / "training/transfer"),
                "--stable_model_path", str(root / "training/transfer/isolated_stable_checkpoint.pth"),
                "--stage1_epochs", str(args.stage1_epochs), "--stage2_epochs", str(args.stage2_epochs)]
    scratch = [sys.executable, str(ROOT / "src/Co-transfer/baselines_Co/train_co_from_scratch.py"), *common,
               "--schema_checkpoint", str(source), "--output_dir", str(root / "training/co_only"),
               "--epochs", str(args.co_only_epochs)]
    if args.cpu:
        transfer.append("--cpu")
        scratch.append("--cpu")
    return {"transfer": transfer, "co_only": scratch}


def run_process(command, args, name):
    environment = os.environ.copy()
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                        "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(args.output_dir / ".matplotlib")})
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        environment[variable] = str(args.native_threads)
    log_path = args.output_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nStarted {datetime.now(timezone.utc).isoformat()}\n")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def validate_trained_checkpoint(metadata, output, model):

    import numpy as np
    import pandas as pd
    output = Path(output)
    development = pd.read_csv(output / "split/development.csv")
    if Path(metadata.get("data_path", "")).resolve() != (output / "split/development.csv").resolve():
        raise ValueError("Checkpoint was not trained on this frozen 125-row development file.")
    train = np.asarray(metadata.get("train_indices", []), dtype=int)
    validation = np.asarray(metadata.get("validation_indices", []), dtype=int)
    if set(train) & set(validation) or set(train) | set(validation) != set(range(len(development))):
        raise ValueError("Selected checkpoint train/validation coverage is invalid.")
    audit = pd.read_csv(output / "split/development_folds.csv")
    expected_validation = np.flatnonzero(audit.fold.to_numpy() == int(metadata["cv_fold"]))
    np.testing.assert_array_equal(np.sort(validation), expected_validation)
    if model == "Transfer_MCGNN":
        if Path(metadata.get("source_checkpoint", "")).resolve() != (output / "inputs/ir_source_checkpoint.pth").resolve():
            raise ValueError("Transfer source differs from the frozen Ir checkpoint.")
    elif metadata.get("source_weights_loaded") is not False:
        raise ValueError("Co-only baseline must not load Ir weight tensors.")


def evaluate_neural(checkpoint_path, args, manifest, model, audit_name=None, oof_name=None):
    import numpy as np
    import pandas as pd
    import torch
    from src.prediction import load_regression_checkpoint, predict_dataframe

    verify_split(args.output_dir, manifest)
    development = pd.read_csv(args.output_dir / "split/development.csv")
    test = pd.read_csv(args.output_dir / "split/test.csv")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    network, metadata, _ = load_regression_checkpoint(str(checkpoint_path), device)
    if model != "Ir_zero_shot_GNN":
        validate_trained_checkpoint(metadata, args.output_dir, model)
        audit = pd.read_csv(Path(checkpoint_path).parent / audit_name).set_index("Sample Name")
        expected = pd.read_csv(args.output_dir / "split/development_folds.csv").set_index("Name")
        if set(audit.index) != set(expected.index):
            raise ValueError("Graph training dropped development samples.")
        np.testing.assert_array_equal(audit.loc[expected.index, "Fold"], expected.fold)
        oof = pd.read_csv(Path(checkpoint_path).parent / oof_name)
        write_results(oof, development.Name.tolist(), development[TARGET].to_numpy(),
                      args.output_dir / "evaluations" / model / "development_oof", model, "development_oof", not args.skip_plots)

    forbidden = [c for c in test.columns if c == TARGET or c.startswith(("True ", "Predicted ", "Class "))]
    unlabeled = test.drop(columns=forbidden)
    with torch.no_grad():
        predictions, _ = predict_dataframe(network, unlabeled, device, batch_size=args.batch_size,
                                             default_catalyst="Co", target_column=TARGET,
                                             xtb_root=str(args.xtb_root.resolve()))
    evaluation = "holdout_zero_shot" if model == "Ir_zero_shot_GNN" else "holdout_test"
    write_results(predictions, test.Name.tolist(), test[TARGET].to_numpy(),
                  args.output_dir / "evaluations" / model / evaluation, model, evaluation, not args.skip_plots)
    dump(args.output_dir / "evaluations" / model / "model_provenance.json",
         {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
          "split_sha256": sha256(args.output_dir / "split/manifest.json"),
          "selected_development_fold": metadata.get("cv_fold") if model != "Ir_zero_shot_GNN" else None,
          "test_labels_used_for_selection": False})
    del network
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_stage(args, name, action, artifact_roots):
    receipt_path = args.output_dir / "completion" / f"{name}.json"
    if receipt_path.exists():
        if not args.resume:
            raise FileExistsError(f"Stage {name} is complete. Use --resume to verify and reuse it.")
        receipt = read_json(receipt_path)
        for relative, expected in receipt["files"].items():
            if sha256(args.output_dir / relative) != expected:
                raise ValueError(f"Completed stage artifact changed: {relative}")
        print(f"[resume] verified and reused {name}", flush=True)
        return
    action()
    files = {str(path.relative_to(args.output_dir)): sha256(path)
             for directory in artifact_roots for path in Path(directory).rglob("*")
             if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts)}
    if not files:
        raise RuntimeError(f"Stage {name} did not produce any artifacts.")
    dump(receipt_path, {"stage": name, "files": files, "completed_utc": datetime.now(timezone.utc).isoformat()})


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.native_threads))
    sys.path.insert(0, str(ROOT / "src/Co-transfer"))
    sys.path.insert(0, str(ROOT / "src/Co-transfer/Cotrans"))
    from fixed_protocol import prepare_frozen_split
    manifest = prepare_frozen_split(args.output_dir, args.data)
    source, source_info = freeze_source(args)
    commands = training_commands(args, source)
    dump(args.output_dir / "training_plan.json", commands)
    print(f"Baseline results: {args.output_dir}")
    if args.prepare_only:
        print("Preparation complete.")
        return
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"prepare_only", "resume", "components", "skip_plots"}}
    settings.update({"source_checkpoint_sha256": source_info["sha256"],
                     "split_manifest_sha256": sha256(args.output_dir / "split/manifest.json"),
                     "additional_input_sha256": {
                         str(p.resolve()): sha256(p) for p in [
                             ROOT / "data/ir_holdout/Database Ir Development.csv",
                             args.xtb_root.resolve() / "xtb_unique_molecule_features.csv",
                             *(ROOT / "outputs/baselines/feature_cache").glob("*_features.npz"),
                             *(ROOT / "outputs/baselines/feature_cache").glob("*_features.json"),
                         ]},
                     "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                     [*Path(__file__).parent.glob("*.py"), ROOT / "src/Co-transfer/transfer_to_co.py",
                                      ROOT / "src/Co-transfer/baselines_Co/train_co_from_scratch.py", ROOT / "baselines/ML/train_rdkit_baselines.py",
                                      ROOT / "baselines/ML/train_random_forest_transfer.py", *Path(ROOT / "src").glob("*.py")]}})
    run_manifest = args.output_dir / "run_settings.json"
    if run_manifest.exists():
        if not (read_json(run_manifest) == settings):
            raise ValueError("Run settings/code/data changed. Use a new isolated output directory.")
        if not args.resume:
            raise FileExistsError("This experiment has already started. Use --resume for the same settings.")
    else:
        dump(run_manifest, settings)
    try:
        if args.components in {"all", "neural"}:
            for stage, model, checkpoint, audit, oof in (
                ("transfer", "Transfer_MCGNN", "best_co_xtb_metal_transfer_regression_checkpoint.pth", "co_cv_split_audit.csv", "transfer_five_fold_oof_predictions.csv"),
                ("co_only", "Co_only_GNN", "best_co_only_from_scratch_checkpoint.pth", "co_only_cv_split_audit.csv", "co_only_oof_predictions.csv"),
            ):
                directory = args.output_dir / "training" / stage
                def action(stage=stage, model=model, directory=directory, checkpoint=checkpoint, audit=audit, oof=oof):
                    run_process(commands[stage], args, stage)
                    evaluate_neural(directory / checkpoint, args, manifest, model, audit, oof)
                run_stage(args, stage, action, [directory, args.output_dir / "evaluations" / model])
            run_stage(args, "zero_shot", lambda: evaluate_neural(source, args, manifest, "Ir_zero_shot_GNN"),
                      [args.output_dir / "evaluations/Ir_zero_shot_GNN"])
        if args.components in {"all", "baselines"}:
            from co_transfer_holdout.classical import run_baselines
            from baselines.ML.train_rdkit_baselines import MODEL_ORDER
            models = [*MODEL_ORDER, "WeightedPooled_RF", "IrResidualCo_RF", "IrOnly_ZeroShot_RF"]
            run_stage(args, "classical", lambda: run_baselines(args, manifest),
                      [args.output_dir / "training/classical", *[args.output_dir / "evaluations" / name for name in models]])
    finally:
        summary = combine_reports(args.output_dir)
        if summary is not None:
            print(summary.loc[summary.evaluation.str.startswith("holdout"),
                              ["model", "evaluation", "n", "correct", "accuracy", "g_mae", "g_rmse", "g_r2"]].to_string(index=False))
    dump(args.output_dir / "requested_components_complete.json", {"components": args.components, "smoke": args.smoke,
                                                                 "finished_utc": datetime.now(timezone.utc).isoformat()})
    print("Finished.")


if __name__ == "__main__":
    main()
