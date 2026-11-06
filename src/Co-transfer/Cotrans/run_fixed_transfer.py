from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys

THIS_DIR = Path(__file__).resolve().parent
CLASS_DIR = THIS_DIR.parent
ROOT = CLASS_DIR.parent.parent
JOINT_DIR = THIS_DIR
for location in (ROOT, ROOT, CLASS_DIR, JOINT_DIR, THIS_DIR):
    sys.path.insert(0, str(location))


def parse_args(argv=None, *, configure=None, output_resolver=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "models/Ir checkpoints.pth")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--split-seed", type=int, choices=[126], default=126)
    p.add_argument("--cv-seed", type=int, choices=[10], default=10)
    p.add_argument("--data", type=Path, default=ROOT / "data/Database Co.csv")
    p.add_argument("--xtb-root", type=Path, default=ROOT / "data" / "xtb_Co_features")
    p.add_argument("--training-seed", type=int, choices=[666], default=666)
    p.add_argument("--stage1-epochs", type=int, default=100)
    p.add_argument("--stage2-epochs", type=int, default=150)
    p.add_argument("--stage1-lr", type=float, default=1e-4)
    p.add_argument("--stage2-lr", type=float, default=3e-5)
    p.add_argument("--stage2-strategy", choices=["global_projection", "full_encoders"], default="global_projection")
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--loss-temperature", type=float, default=0.5, help="Ordinal smoothing scale (kcal/mol), not physical temperature")
    p.add_argument("--classification-weight", type=float, default=0.5, help="Fixed ordinal penalty weight; choose using development only")
    p.add_argument("--regression-scale", type=float, default=1.0, help="Fixed G-error scale in kcal/mol")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--native-threads", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--smoke", action="store_true", help="One epoch per stage with a separate output directory")
    p.add_argument("--skip-plots", action="store_true")
    if configure is not None:
        configure(p)
    args = p.parse_args(argv)
    import math
    if min(args.stage1_epochs, args.stage2_epochs, args.batch_size, args.patience, args.native_threads) < 1 or args.scheduler_patience < 0 or args.warmup_epochs < 0:
        p.error("Epoch/batch/patience/thread settings must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.stage1_lr, args.stage2_lr, args.loss_temperature, args.regression_scale)):
        p.error("Learning rates and loss temperature must be finite and positive")
    if not 0 <= args.dropout < 1 or any(not math.isfinite(v) or v < 0 for v in (args.weight_decay, args.classification_weight)):
        p.error("Invalid dropout or nonnegative weight setting")
    if not all(0 <= seed < 2**32 for seed in (args.split_seed, args.cv_seed, args.training_seed)):
        p.error("Seeds must be in [0, 2**32)")
    args.mode = "holdout"
    args.data = args.data.resolve()
    if args.prepare_only and args.cache_only:
        p.error("Choose prepare-only or cache-only")
    if args.smoke:
        args.stage1_epochs = args.stage2_epochs = 1
    from experiment_protocol import output_path
    args.output_dir = (output_resolver(args) if output_resolver is not None else
                       output_path(args.output_dir or ROOT / "outputs/co_transfer" / ("smoke_fixed_seed_126_joint" if args.smoke else "fixed_seed_126_joint"), args.smoke))
    args.checkpoint = args.checkpoint.resolve()
    args.xtb_root = args.xtb_root.resolve()
    return args


def main(argv=None, *, experiment=None):
    args = parse_args(argv) if experiment is None else experiment.parse_args(argv)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / ".matplotlib")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.native_threads)

    import transfer_to_co as original
    import numpy as np
    import pandas as pd
    import torch
    from co_transfer_holdout.protocol import dump, read_json, sha256, combine_reports
    from src.models import build_regression_model_from_checkpoint
    from src.dataset import build_dataset, filter_valid_indices
    from src.prediction import load_regression_checkpoint, predict_dataframe
    from src.train_eval import predict_regressor
    from src.xtb_features import get_xtb_store
    from joint_objective import JointClassificationRegressionLoss as OrdinalClassificationLoss, selection_key
    from joint_training import train_joint as train_classification, score_predictions, cpu_state
    from joint_protocol import assert_checkpoint_protocol
    from fixed_protocol import prepare_inputs, validate_tables, verify_split, write_results, validate_source_data
    from fixed_graph_cache import load_cache, select_samples, cache_identity, build_or_load
    from experiment_protocol import (unlabelled_test,
                                     write_receipt, verify_receipt, register_run_settings)
    if experiment is not None:
        prepare_inputs = experiment.prepare_inputs
        assert_checkpoint_protocol = experiment.assert_checkpoint_protocol

    torch.set_num_threads(args.native_threads)
    full_frame = validate_source_data(args.data)
    initial_state, initial_metadata, _ = original.read_regression_checkpoint(str(args.checkpoint))
    graph_settings = original.feature_settings_from_checkpoint(initial_state, initial_metadata)
    del initial_state
    args.cache_key, _ = cache_identity(full_frame, graph_settings, args.xtb_root)
    args.cache_dir = ROOT / "outputs/07_runtime_cache/co_graph_cache" / args.cache_key
    development, test, audit = prepare_inputs(args)
    output = args.output_dir
    frozen_inputs = read_json(output / "frozen_inputs.json")
    code_paths = sorted([*THIS_DIR.glob("*.py"), CLASS_DIR / "experiment_protocol.py",
                         ROOT / "baselines/ML/train_rdkit_baselines.py", ROOT / "src/Co-transfer/co_transfer_holdout/run_co_holdout.py", *list((ROOT / "src").glob("*.py")),
                         ROOT / "src/Co-transfer/transfer_to_co.py", ROOT / "src/Co-transfer/co_transfer_holdout/protocol.py"])
    if experiment is not None:
        code_paths.extend(experiment.code_paths())
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in ("prepare_only", "cache_only", "resume", "skip_plots", "development_only")}
    settings["input_hashes"] = frozen_inputs
    settings["code_hashes"] = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    settings["selection"] = "validation joint loss asc, ACC desc, macro F1 desc"
    settings["loss"] = "MSE((predicted G - true G)/regression_scale) + classification_weight * ordinal_BCE"
    settings["experiment_role"] = "Fixed Co development and test partitions"
    if experiment is not None:
        settings["experiment_role"] = experiment.ROLE
    register_run_settings(output, settings)
    if args.prepare_only:
        print(f"Prepared {len(development)} development / {len(test)} test records and five frozen folds: {output}")
        return
    if (output / "complete.json").exists():
        if not args.resume:
            raise FileExistsError("Run already complete; use --resume to verify it or a new output directory")
        verify_receipt(output)
        print(f"Existing completed {'SMOKE' if args.smoke else 'formal'} run verified: {output}")
        return
    if (output / "training").exists() and not args.resume:
        raise FileExistsError("Training artifacts exist. Use --resume with unchanged settings.")

    build_or_load(full_frame, graph_settings, args.xtb_root, ROOT / "outputs/07_runtime_cache/co_graph_cache")
    if args.cache_only:
        print(f"Graph cache ready: {args.cache_dir}")
        return
    device = torch.device("cpu") if args.cpu else original.get_device(mem_fraction=0.7)
    source_path = output / "inputs/ir_source_checkpoint.pth"
    state, source_metadata, source_config = original.read_regression_checkpoint(str(source_path))
    original.validate_feature_schema(source_metadata)
    feature_settings = original.feature_settings_from_checkpoint(state, source_metadata)
    full_data = pd.read_csv(args.data)
    full_dataset, cache_record = load_cache(args.cache_dir, args.cache_key, full_data)
    dataset = select_samples(full_dataset, development)
    global_dim = cache_record["global_dim"]
    if len(dataset) != len(development) or [str(row[0]) for row in dataset] != development.Name.astype(str).tolist():
        raise ValueError("Cached development graph alignment failed")
    print(f"Loaded cached graphs: {len(full_dataset)} total, {len(dataset)} development, {len(test)} test", flush=True)
    if global_dim != source_config["global_feature_dim"] or dataset[0][1].ndata["global"].shape[1] != source_config["environment_global_feature_dim"]:
        raise ValueError("Source/Co graph feature dimensions differ")
    actual_groups = original.reaction_group_ids(development, development.Name.tolist())
    np.testing.assert_array_equal(actual_groups, audit.set_index("Name").loc[development.Name, "reaction_group_id"])
    fold_ids = validate_tables(development, test, audit)
    dump(output / "xtb_feature_coverage.json", cache_record["coverage"])

    audit.to_csv(output / "training_fold_assignments.csv", index=False)
    objective = OrdinalClassificationLoss(args.loss_temperature, args.classification_weight, args.regression_scale)
    fold_summaries, oof_tables = [], []
    for fold in range(1, 6):
        folder = output / "training" / f"fold_{fold}"
        checkpoint_path = folder / "joint_checkpoint.pth"
        train_indices, validation_indices = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        if args.resume and (folder / "complete.json").exists():
            verify_receipt(folder)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert_checkpoint_protocol(checkpoint["metadata"], output, fold)
            fold_summaries.append(read_json(folder / "summary.json"))
            oof_tables.append(pd.read_csv(folder / "validation_predictions.csv"))
            del checkpoint
            print(f"Verified and reused completed fold {fold}", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        original.setup_all_seeds(args.training_seed)
        model, model_config = build_regression_model_from_checkpoint(state, source_metadata)
        model.load_state_dict(state, strict=True)
        model.to(device)
        original.set_transfer_dropout(model, args.dropout)
        train_data = [dataset[i] for i in train_indices]
        validation_data = [dataset[i] for i in validation_indices]
        stage_results, stage_states = {}, {}
        for stage, modules, epochs, lr in (
            ("stage1", original.STAGE1_TRAINABLE_MODULES, args.stage1_epochs, args.stage1_lr),
            ("stage2", original.stage2_trainable_modules(args.stage2_strategy), args.stage2_epochs, args.stage2_lr),
        ):
            original.set_requires_grad(model, modules)
            print(f"Fold {fold}/5 {stage}: {len(train_data)} train / {len(validation_data)} validation; joint loss", flush=True)
            result = train_classification(model, train_data, validation_data, device, epochs=epochs, learning_rate=lr,
                                           batch_size=args.batch_size, weight_decay=args.weight_decay, patience=args.patience,
                                           scheduler_patience=args.scheduler_patience, temperature=args.loss_temperature,
                                           classification_weight=args.classification_weight, regression_scale=args.regression_scale,
                                           warmup_epochs=args.warmup_epochs,
                                           epoch_offset=0 if stage == "stage1" else len(stage_results["stage1"]["history"]))
            stage_results[stage] = result
            stage_states[stage] = cpu_state(model)
            pd.DataFrame(result["history"]).to_csv(folder / f"{stage}_history.csv", index=False)
        final_stage = max(stage_results, key=lambda key: selection_key(stage_results[key]["best_metrics"]))
        model.load_state_dict(stage_states[final_stage], strict=True)
        names, targets, predictions, parts = predict_regressor(model, validation_data, device,
                                                              batch_size=args.batch_size, return_components=True)
        values = score_predictions(targets, predictions, objective)
        if selection_key(values) != selection_key(stage_results[final_stage]["best_metrics"]):

            np.testing.assert_allclose(selection_key(values), selection_key(stage_results[final_stage]["best_metrics"]), rtol=0, atol=1e-7)
        validation = pd.DataFrame({"Name": names, "Fold": fold, "Dataset Position": validation_indices,
                                   "True G (kcal/mol)": targets, "Predicted G (kcal/mol)": predictions,
                                   "Temperature (K)": parts["temperature_kelvin"],
                                   "Predicted Delta H (kcal/mol)": parts["delta_h_kcal_mol"],
                                   "Predicted Delta S (cal/mol/K)": parts["delta_s_cal_mol_k"], "status": "OK"})
        metadata = deepcopy(source_metadata)
        metadata.update({"task": "co_transfer_joint_regression_classification", "default_catalyst": "Co",
                         "model_config": model_config, "source_checkpoint": str(source_path),
                         "source_checkpoint_sha256": frozen_inputs["source_checkpoint_sha256"],
                         "data_path": str(output / "split/development.csv"), "cv_fold": fold,
                         "train_indices": train_indices.tolist(), "validation_indices": validation_indices.tolist(),
                         "split_manifest_sha256": frozen_inputs["split_manifest_sha256"],
                         "classification_boundaries": [24.0, 26.0], "training_config": settings,
                         "feature_settings": feature_settings, "xtb_data_provenance": cache_record["coverage"],
                         "final_stage": final_stage, "best_epoch": stage_results[final_stage]["best_epoch"],
                         "validation_method": "fixed_development_grouped_5fold_with_joint_loss_early_stopping",
                         "selection_metric": settings["selection"], "metrics": values,
                         "test_labels_used_for_selection": False, "test_used_for_seed_search": False, "historical_split_was_search_selected": True,
                         "split_seed": args.split_seed, "evaluation_mode": args.mode, "graph_cache_key": args.cache_key, "smoke": args.smoke})
        metadata["thermodynamic_relation"] = {**metadata["thermodynamic_relation"], "loss": settings["loss"]}
        assert_checkpoint_protocol(metadata, output, fold)
        torch.save({"model_state_dict": cpu_state(model), "metadata": metadata}, checkpoint_path)
        summary = {"fold": fold, "train_size": len(train_data), "validation_size": len(validation_data),
                   "final_stage": final_stage, "best_epoch": metadata["best_epoch"],
                   "stage_best": {key: {k: v for k, v in result.items() if k != "history"} for key, result in stage_results.items()},
                   "validation": values, "checkpoint": str(checkpoint_path)}
        dump(folder / "summary.json", summary)
        validation.to_csv(folder / "validation_predictions.csv", index=False)
        write_receipt(folder, ["joint_checkpoint.pth", "summary.json", "validation_predictions.csv", "stage1_history.csv", "stage2_history.csv"])
        fold_summaries.append(summary)
        oof_tables.append(validation)
        del model, stage_states, train_data, validation_data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(oof_tables, ignore_index=True).sort_values("Dataset Position")
    np.testing.assert_array_equal(oof.Name, development.Name)
    np.testing.assert_array_equal(oof.Fold, fold_ids)
    model_name = "Co_transfer_joint"
    oof_role = "development_oof" if args.mode == "holdout" else "full_oof"
    oof_dir = output / "evaluations" / model_name / oof_role
    _, oof_score = write_results(oof, development.Name.tolist(), development["G (kcal/mol)"].to_numpy(),
                                 oof_dir, model_name, oof_role, plots=not args.skip_plots)
    pd.DataFrame([{"fold": s["fold"], "train_size": s["train_size"], "final_stage": s["final_stage"],
                   "best_epoch": s["best_epoch"], **s["validation"]} for s in fold_summaries]).to_csv(output / "fold_metrics.csv", index=False)
    artifacts = ["run_settings.json", "fold_metrics.csv", "summary_metrics.csv", "all_predictions.csv",
                 "loss_components_summary.csv", "frozen_inputs.json", "split/manifest.json",
                 "split/development.csv", "split/development_folds.csv",
                 str(oof_dir.relative_to(output) / "predictions.csv"), str(oof_dir.relative_to(output) / "metrics.json")]
    evaluations = [(oof_role, oof_dir)]
    if args.mode == "holdout":
        selected = max(fold_summaries, key=lambda item: selection_key(item["validation"]))
        selected_path = output / "best_joint_checkpoint.pth"
        shutil.copy2(selected["checkpoint"], selected_path)
        dump(output / "selected_model.json", {"fold": selected["fold"], "stage": selected["final_stage"],
             "validation": selected["validation"], "selection": settings["selection"],
             "checkpoint_sha256": sha256(selected_path), "test_used_for_selection": False})
        verify_split(output)
        network, metadata, _ = load_regression_checkpoint(str(selected_path), device)
        assert_checkpoint_protocol(metadata, output, selected["fold"])

        heldout = select_samples(full_dataset, test, hide_targets=True)
        names, _, predicted, parts = predict_regressor(network, heldout, device, batch_size=args.batch_size, return_components=True)
        predictions = pd.DataFrame({"Name": names, "Predicted G (kcal/mol)": predicted,
                    "Temperature (K)": parts["temperature_kelvin"], "Predicted Delta H (kcal/mol)": parts["delta_h_kcal_mol"],
                    "Predicted Delta S (cal/mol/K)": parts["delta_s_cal_mol_k"], "status": "OK"})
        test_dir = output / "evaluations" / model_name / "holdout_test"
        _, test_score = write_results(predictions, test.Name.tolist(), test["G (kcal/mol)"].to_numpy(),
                                     test_dir, model_name, "holdout_test", plots=not args.skip_plots)
        evaluations.append(("holdout_test", test_dir))
        artifacts.extend(["best_joint_checkpoint.pth", "selected_model.json", "split/test.csv",
                          str(test_dir.relative_to(output) / "predictions.csv"), str(test_dir.relative_to(output) / "metrics.json")])
        print(f"Test ACC={test_score['accuracy']:.4%}, n={len(test)}", flush=True)

    combine_reports(output)
    penalties = []
    for evaluation, directory in evaluations:
        table = pd.read_csv(directory / "predictions.csv")
        penalties.append({"evaluation": evaluation, **score_predictions(table["True G (kcal/mol)"].to_numpy(),
                                    table["Predicted G (kcal/mol)"].to_numpy(), objective)})
    pd.DataFrame(penalties).to_csv(output / "loss_components_summary.csv", index=False)
    write_receipt(output, artifacts, {"smoke": args.smoke, "mode": args.mode, "seed": args.split_seed,
                                     "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(f"{args.mode} complete: OOF ACC={oof_score['accuracy']:.4%}, n={len(development)}. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
