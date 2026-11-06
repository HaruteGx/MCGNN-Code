import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "Co-transfer"))
SRC_DIR = PROJECT_DIR
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
normalized_src = os.path.normcase(os.path.abspath(SRC_DIR))
sys.path[:] = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or os.curdir)) != normalized_src
]
sys.path.insert(0, SRC_DIR)
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

from transfer_to_co import (
    N_SPLITS,
    PHYSICS_HEAD,
    REACTION_GROUP_COLUMNS,
    TARGET_COLUMN,
    THERMODYNAMIC_RELATION,
    copy_state_to_cpu,
    feature_settings_from_checkpoint,
    make_cv_splits,
    publication_sources,
    reaction_group_ids,
    regression_strata,
    setup_all_seeds,
    source_target_strata,
    take,
    validate_feature_schema,
    validate_thermodynamic_checkpoint,
)
from src.config import get_device
from src.dataset import (
    REACTION_SITE_INDEX_FEATURE_LAYOUT,
    build_dataset,
    filter_valid_indices,
)
from src.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from src.ir_protocol import ir_run_paths
from src.models import (
    ReactionEnergyPredictionModel,
    build_regression_model_from_checkpoint,
)
from src.train_eval import (
    evaluate_and_save,
    plot_training_history,
    regression_metrics,
    save_regression_scatter,
    train_regressor,
)
from src.xtb_features import get_xtb_store
from src.regression_classification import export_classification


PROTOCOL_PATHS = ir_run_paths(PROJECT_DIR)
DEFAULT_SCHEMA_CHECKPOINT = str(PROTOCOL_PATHS["stable_checkpoint"])
DEFAULT_DATA = os.path.join(PROJECT_DIR, "data", "Database Co.csv")
DEFAULT_XTB_ROOT = os.path.join(PROJECT_DIR, "data", "xtb_Co_features")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "baselines_Co_old",
    "full_oof_seed_10",
    "co_only_from_scratch",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path):
    extension = os.path.splitext(path)[1].lower()
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if extension == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported Co data file: {path}")


def read_schema_checkpoint(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Ir schema checkpoint not found. Train the current Ir model first: "
            f"{path}"
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The schema checkpoint must include state dict and metadata.")
    state_dict = checkpoint["model_state_dict"]
    metadata = checkpoint.get("metadata", {}) or {}
    validate_thermodynamic_checkpoint(state_dict, metadata)
    schema_model, model_config = build_regression_model_from_checkpoint(
        state_dict, metadata
    )


    del schema_model
    return state_dict, metadata, model_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a randomly initialized Delta-H/Delta-S model on Co only."
    )
    parser.add_argument("--schema_checkpoint", default=DEFAULT_SCHEMA_CHECKPOINT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--xtb_root", default=DEFAULT_XTB_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cv_strategy",
        choices=("group_source_target", "source_target", "target_only"),
        default="group_source_target",
    )
    parser.add_argument("--cv_seed", type=int, default=10)
    parser.add_argument("--training_seed", type=int, default=666)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--scheduler_factor", type=float, default=0.50)
    parser.add_argument("--minimum_learning_rate", type=float, default=1e-8)
    parser.add_argument("--mem_fraction", type=float, default=0.7)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _fold_checkpoint(
    model,
    source_metadata,
    model_config,
    feature_settings,
    xtb_report,
    args,
    fold,
    train_indices,
    validation_indices,
    valid_source_rows,
    reaction_groups,
    metrics,
    best_epoch,
):
    train_group_ids = sorted(set(reaction_groups[train_indices]))
    validation_group_ids = sorted(set(reaction_groups[validation_indices]))
    overlap = sorted(set(train_group_ids).intersection(validation_group_ids))
    if args.cv_strategy == "group_source_target" and overlap:
        raise RuntimeError(f"Reaction groups cross fold {fold}: {overlap}")
    return {
        "model_state_dict": copy_state_to_cpu(model),
        "metadata": {
            "task": "co_only_random_initialization_regression",
            "purpose": "Co-only baseline; no Ir weights loaded",
            "target_column": TARGET_COLUMN,
            "output_dimension": 1,
            "latent_output_dimension": 2,
            "head": PHYSICS_HEAD,
            "thermodynamic_relation": dict(THERMODYNAMIC_RELATION),
            "initialization": "random_per_fold",
            "training_labels": "Co only",
            "source_checkpoint_role": "architecture_and_feature_schema_only",
            "source_weights_loaded": False,
            "schema_checkpoint": os.path.abspath(args.schema_checkpoint),
            "schema_checkpoint_sha256": sha256_file(args.schema_checkpoint),
            "schema_checkpoint_task": source_metadata.get("task"),
            "data_path": os.path.abspath(args.data),
            "data_sha256": sha256_file(args.data),
            "xtb_data_provenance": xtb_report,
            "fingerprint_config": dict(
                source_metadata.get("fingerprint_config", {}) or {}
            ),
            "global_descriptor_config": dict(
                source_metadata.get("global_descriptor_config", {}) or {}
            ),
            "xtb_feature_config": dict(source_metadata["xtb_feature_config"]),
            "reaction_site_index_features": REACTION_SITE_INDEX_FEATURE_LAYOUT,
            "node_feature_schema": list(NODE_FEATURE_NAMES),
            "edge_feature_schema": list(EDGE_FEATURE_NAMES),
            "model_config": dict(model_config),
            "feature_settings": dict(feature_settings),
            "default_catalyst": "Co",
            "training_config": {
                "loss": "MSELoss on derived Delta G",
                "optimizer": "AdamW",
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "early_stopping_patience": int(args.patience),
                "all_model_parameters_trainable": True,
            },
            "n_splits": N_SPLITS,
            "cv_strategy": args.cv_strategy,
            "cv_seed": int(args.cv_seed),
            "training_seed": int(args.training_seed),
            "cv_fold": int(fold),
            "best_epoch": int(best_epoch),
            "train_indices": [int(index) for index in train_indices],
            "validation_indices": [int(index) for index in validation_indices],
            "valid_source_rows": [int(index) for index in valid_source_rows],
            "reaction_grouping": {
                "columns": list(REACTION_GROUP_COLUMNS),
                "train_group_ids": train_group_ids,
                "validation_group_ids": validation_group_ids,
                "overlap_group_ids": overlap,
            },
            "metrics": metrics,
        },
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = (
        torch.device("cpu")
        if args.cpu
        else get_device(mem_fraction=args.mem_fraction)
    )
    setup_all_seeds(args.training_seed)

    source_state, source_metadata, model_config = read_schema_checkpoint(
        args.schema_checkpoint
    )
    validate_feature_schema(source_metadata)
    feature_settings = feature_settings_from_checkpoint(
        source_state, source_metadata
    )


    del source_state

    data = load_table(args.data)
    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"Co-only data is missing {TARGET_COLUMN!r}.")
    xtb_store = (
        get_xtb_store(args.xtb_root)
        if feature_settings["use_xtb_features"]
        else None
    )
    valid_source_rows = filter_valid_indices(data, default_catalyst="Co")
    dataset, global_feature_dim = build_dataset(
        data,
        valid_source_rows,
        torch.device("cpu"),
        target_column=TARGET_COLUMN,
        default_catalyst="Co",
        xtb_store=xtb_store,
        **feature_settings,
    )
    if len(dataset) != len(valid_source_rows):
        raise RuntimeError(
            "Some validated Co rows failed graph construction; source alignment "
            "cannot be guaranteed."
        )
    if len(dataset) < N_SPLITS:
        raise RuntimeError(f"Need at least {N_SPLITS} valid Co rows.")
    if int(model_config["global_feature_dim"]) != int(global_feature_dim):
        raise ValueError("Co global feature width differs from the schema checkpoint.")
    environment_global_dim = int(dataset[0][1].ndata["global"].shape[1])
    if int(model_config["environment_global_feature_dim"]) != environment_global_dim:
        raise ValueError("Co environment feature width differs from the schema checkpoint.")

    targets = np.asarray([float(sample[-1]) for sample in dataset], dtype=float)
    sample_names = [sample[0] for sample in dataset]
    if args.cv_strategy in {"group_source_target", "source_target"}:
        strata, stratum_edges, split_audit = source_target_strata(
            targets, sample_names
        )
    else:
        strata, stratum_edges = regression_strata(targets)
        sources = publication_sources(sample_names)
        split_audit = {
            "strategy": "target_quantiles_only",
            "sources": sources.tolist(),
            "source_buckets": sources.tolist(),
            "strata": [str(value) for value in strata],
        }
    reaction_groups = reaction_group_ids(data, sample_names)
    cv_splits = make_cv_splits(
        args.cv_strategy, strata, reaction_groups, args.cv_seed
    )
    fold_assignments = np.zeros(len(dataset), dtype=int)
    for fold, (_, validation_indices) in enumerate(cv_splits, start=1):
        fold_assignments[validation_indices] = fold
    split_rows = pd.DataFrame(
        {
            "Dataset Position": np.arange(len(dataset)),
            "Source Row Index": valid_source_rows,
            "Sample Name": sample_names,
            "Reaction Group ID": reaction_groups,
            "Fold": fold_assignments,
            "Source": split_audit["sources"],
            "Source Bucket": split_audit["source_buckets"],
            "Stratum": split_audit["strata"],
            TARGET_COLUMN: targets,
        }
    )
    overlaps = (
        split_rows.groupby("Reaction Group ID")["Fold"]
        .nunique()
        .loc[lambda values: values > 1]
        .index.tolist()
    )
    if args.cv_strategy == "group_source_target" and overlaps:
        raise RuntimeError(f"Grouped Co-only folds overlap: {overlaps}")
    split_rows.to_csv(
        os.path.join(args.output_dir, "co_only_cv_split_audit.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    xtb_report = xtb_store.report() if xtb_store is not None else None
    with open(
        os.path.join(args.output_dir, "co_only_protocol.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            {
                "task": "co_only_random_initialization_regression",
                "source_weights_loaded": False,
                "co_rows": int(len(dataset)),
                "cv_strategy": args.cv_strategy,
                "reaction_group_overlap": overlaps,
                "stratification_edges": stratum_edges.tolist(),
                "xtb_data_provenance": xtb_report,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    fold_rows = []
    oof_frames = []
    best_score = (-float("inf"), -float("inf"))
    best_model_path = None
    for fold, (train_indices, validation_indices) in enumerate(
        cv_splits, start=1
    ):
        setup_all_seeds(args.training_seed)
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        train_data = take(dataset, train_indices)
        validation_data = take(dataset, validation_indices)
        print(
            f"\n===== Co-only from-scratch fold {fold}/{N_SPLITS} =====\n"
            f"Train/Validation: {len(train_data)}/{len(validation_data)}"
        )


        model = ReactionEnergyPredictionModel(**model_config).to(device)
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Every Co-only model parameter must be trainable.")
        history = train_regressor(
            model,
            train_data,
            validation_data,
            device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            minimum_learning_rate=args.minimum_learning_rate,
            early_stopping_patience=args.patience,
            restore_best_weights=True,
        )
        plot_training_history(
            history, os.path.join(fold_dir, "training_history.png")
        )
        pd.DataFrame(
            {
                "epoch": np.arange(1, len(history["train_loss"]) + 1),
                "train_mse": history["train_loss"],
                "validation_r2": history["validation_r2"],
                "validation_rmse": history["validation_rmse"],
                "validation_mae": history["validation_mae"],
                "learning_rate": history["learning_rate"],
            }
        ).to_csv(os.path.join(fold_dir, "training_history.csv"), index=False)

        train_metrics, _ = evaluate_and_save(
            model,
            train_data,
            device,
            os.path.join(fold_dir, "train_results.xlsx"),
            f"Co-only Fold {fold} Train",
            batch_size=args.batch_size,
            target_column=TARGET_COLUMN,
            classification_dataset="Co",
        )
        validation_metrics, validation_results = evaluate_and_save(
            model,
            validation_data,
            device,
            os.path.join(fold_dir, "validation_results.xlsx"),
            f"Co-only Fold {fold} Validation",
            batch_size=args.batch_size,
            target_column=TARGET_COLUMN,
            classification_dataset="Co",
        )
        oof = validation_results.copy()
        oof.insert(0, "Dataset Position", validation_indices.astype(int))
        oof.insert(1, "Fold", int(fold))
        oof_frames.append(oof)
        fold_row = {
            "fold": int(fold),
            "train_size": len(train_data),
            "validation_size": len(validation_data),
            "epochs_run": len(history["train_loss"]),
            "best_epoch": int(history["best_epoch"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        fold_rows.append(fold_row)
        checkpoint = _fold_checkpoint(
            model,
            source_metadata,
            model_config,
            feature_settings,
            xtb_report,
            args,
            fold,
            train_indices,
            validation_indices,
            valid_source_rows,
            reaction_groups,
            {"train": train_metrics, "validation": validation_metrics},
            history["best_epoch"],
        )
        model_path = os.path.join(
            fold_dir, "co_only_from_scratch_checkpoint.pth"
        )
        torch.save(checkpoint, model_path)
        score = (validation_metrics["r2"], -validation_metrics["rmse"])
        if score > best_score:
            best_score = score
            best_model_path = model_path
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True).sort_values(
        "Dataset Position"
    )
    if len(oof_df) != len(dataset) or oof_df["Dataset Position"].nunique() != len(dataset):
        raise RuntimeError("Every Co row must receive exactly one OOF prediction.")
    true_column = f"True {TARGET_COLUMN}"
    predicted_column = f"Predicted {TARGET_COLUMN}"
    oof_metrics = regression_metrics(oof_df[true_column], oof_df[predicted_column])
    oof_df = export_classification(oof_df, args.output_dir, "Co", stem="oof_classification",
                                   role="Co-only GNN grouped OOF: regression to classes")
    aggregate_rows = []
    for metric_name in ("r2", "rmse", "mae"):
        column = f"validation_{metric_name}"
        aggregate_rows.append(
            {
                "metric": metric_name,
                "fold_mean": float(fold_df[column].mean()),
                "fold_std": float(fold_df[column].std(ddof=1)),
                "oof_value": float(oof_metrics[metric_name]),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_path = os.path.join(args.output_dir, "co_only_summary.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name="Fold Metrics", index=False)
        aggregate_df.to_excel(writer, sheet_name="Mean Std and OOF", index=False)
        oof_df.to_excel(writer, sheet_name="OOF Predictions", index=False)
        split_rows.to_excel(writer, sheet_name="CV Split Audit", index=False)
    oof_df.to_csv(
        os.path.join(args.output_dir, "co_only_oof_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_regression_scatter(
        oof_df[true_column],
        oof_df[predicted_column],
        os.path.join(args.output_dir, "co_only_oof_scatter.png"),
        "Co-only From-scratch Five-fold OOF",
        target_label=TARGET_COLUMN,
    )
    if best_model_path is None:
        raise RuntimeError("No Co-only checkpoint was produced.")
    best_output = os.path.join(
        args.output_dir, "best_co_only_from_scratch_checkpoint.pth"
    )
    shutil.copyfile(best_model_path, best_output)
    print(f"\nCo-only OOF metrics: {oof_metrics}")
    print(f"Summary: {summary_path}")
    print(f"Best validation-fold checkpoint: {best_output}")


if __name__ == "__main__":
    main()
