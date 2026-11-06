import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "Co-transfer"))
SRC_DIR = PROJECT_DIR


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

from src.config import get_device
from src.dataset import (
    REACTION_SITE_INDEX_FEATURE_LAYOUT,
    TEMPERATURE_CELSIUS_COLUMN,
    build_dataset,
    filter_valid_indices,
)
from src.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from src.models import build_regression_model_from_checkpoint
from src.xtb_features import get_xtb_store
from src.ir_protocol import ir_run_paths
from src.train_eval import (
    evaluate_and_save,
    plot_training_history,
    regression_metrics,
    save_regression_scatter,
    train_regressor,
)


TARGET_COLUMN = "G (kcal/mol)"
N_SPLITS = 5
NUM_STRATIFICATION_BINS = 10
PHYSICS_HEAD = "enthalpy_entropy_physics"
THERMODYNAMIC_RELATION = {
    "temperature_source_column": TEMPERATURE_CELSIUS_COLUMN,
    "temperature_model_unit": "K",
    "delta_h_unit": "kcal/mol",
    "delta_s_unit": "cal/mol/K",
    "delta_g_unit": "kcal/mol",
    "formula": "Delta G = Delta H - T * Delta S / 1000",
    "temperature_usage": "parameter-free_physics_layer_only",
    "loss": "MSE(predicted Delta G, measured Delta G)",
}
REACTION_GROUP_COLUMNS = ("Cat", "sub-type", "L-type", "B-type", "P")
STAGE1_TRAINABLE_MODULES = ["catalyst_context", "fc1", "fc2"]
STAGE2_GLOBAL_PROJECTION_MODULES = [
    "reactant_gnn.global_layer",
    "reactant_gnn.gnn_layer.res_fc",
    "product_gnn.global_layer",
    *STAGE1_TRAINABLE_MODULES,
]
STAGE2_FULL_ENCODER_MODULES = [
    "reactant_gnn", "product_gnn", *STAGE1_TRAINABLE_MODULES,
]


def _default_co_data_path():

    return os.path.join(PROJECT_DIR, "data", "Database Co.csv")


DEFAULT_DATA_PATH = _default_co_data_path()
DEFAULT_CHECKPOINT_PATH = str(ir_run_paths(PROJECT_DIR)["best_checkpoint"])
DEFAULT_XTB_ROOT = os.path.join(PROJECT_DIR, "data", "xtb_Co_features")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "transfer_learning_xtb_metal_regression",
    ir_run_paths(PROJECT_DIR)["tag"],
    "grouped_source_balanced_run",
)
STABLE_MODEL_PATH = str(ir_run_paths(PROJECT_DIR)["co_stable_checkpoint"])


def setup_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_table(path):
    extension = os.path.splitext(path)[1].lower()
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if extension == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {path}")


def read_regression_checkpoint(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Ir Delta-G checkpoint not found. Run scripts/train_full_model.py "
            f"or update --checkpoint: {path}"
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint.get("metadata", {}) or {}
    else:
        state_dict = checkpoint
        metadata = {}
    validate_thermodynamic_checkpoint(state_dict, metadata)
    model, model_config = build_regression_model_from_checkpoint(state_dict, metadata)
    model.load_state_dict(state_dict, strict=True)
    return state_dict, metadata, model_config


def validate_thermodynamic_checkpoint(state_dict, metadata):

    if "fc2.weight" not in state_dict or int(state_dict["fc2.weight"].shape[0]) != 2:
        raise ValueError(
            "The source checkpoint does not use the required two-output "
            "Delta-H/Delta-S head. Retrain the Ir model first."
        )
    saved_head = (metadata or {}).get("head")
    if saved_head != PHYSICS_HEAD:
        raise ValueError(
            f"Unsupported source checkpoint head {saved_head!r}; expected "
            f"{PHYSICS_HEAD!r}. Retrain the Ir source model with the current code."
        )
    saved_relation = dict(
        (metadata or {}).get("thermodynamic_relation", {}) or {}
    )
    for key in (
        "temperature_model_unit",
        "delta_h_unit",
        "delta_s_unit",
        "delta_g_unit",
        "formula",
    ):
        if key in saved_relation and saved_relation[key] != THERMODYNAMIC_RELATION[key]:
            raise ValueError(
                f"Source checkpoint thermodynamic metadata differs for {key}: "
                f"{saved_relation[key]!r} vs {THERMODYNAMIC_RELATION[key]!r}."
            )


def validate_feature_schema(metadata):
    saved_nodes = list((metadata or {}).get("node_feature_schema", []) or [])
    saved_edges = list((metadata or {}).get("edge_feature_schema", []) or [])
    if saved_nodes and saved_nodes != list(NODE_FEATURE_NAMES):
        raise ValueError("Checkpoint node features do not match this repository.")
    if saved_edges and saved_edges != list(EDGE_FEATURE_NAMES):
        raise ValueError("Checkpoint edge features do not match this repository.")


def feature_settings_from_checkpoint(state_dict, metadata):
    fingerprint_config = dict(
        (metadata or {}).get("fingerprint_config", {}) or {}
    )
    descriptor_config = dict(
        (metadata or {}).get("global_descriptor_config", {}) or {}
    )
    fingerprint_nbits = fingerprint_config.get("n_bits_per_fingerprint")
    if fingerprint_nbits is None:
        global_dim = int(state_dict["reactant_gnn.global_layer.weight"].shape[1])
        descriptor_count = len(descriptor_config.get("global_descriptor_names", []))
        fingerprint_elements = global_dim - descriptor_count
        if fingerprint_elements <= 0 or fingerprint_elements % 2:
            raise ValueError(
                "Cannot infer fingerprint width from the source checkpoint."
            )
        fingerprint_nbits = fingerprint_elements // 2
    return {
        "use_xtb_features": bool(metadata["xtb_feature_config"]["enabled"]),
        "fp_nbits": int(fingerprint_nbits),
        "include_3d_descriptors": bool(descriptor_config.get("include_3d", False)),
        "descriptor_num_conformers": int(
            descriptor_config.get("max_conformers", 3)
        ),
        "descriptor_conformer_seed": int(
            descriptor_config.get("conformer_seed", 20260822)
        ),
        "optimize_conformers": bool(
            descriptor_config.get("optimize_conformers", True)
        ),
    }


def regression_strata(targets):
    strata, edges = pd.qcut(
        np.asarray(targets, dtype=float),
        q=NUM_STRATIFICATION_BINS,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    strata = np.asarray(strata, dtype=int)
    _, counts = np.unique(strata, return_counts=True)
    if len(counts) < 2 or int(counts.min()) < N_SPLITS:
        raise ValueError(
            f"Five-fold transfer needs at least {N_SPLITS} samples in every "
            f"target-quantile bin; bin counts are {counts.tolist()}."
        )
    return strata, np.asarray(edges, dtype=float)


def publication_sources(names):

    sources = []
    for name in names:
        text = str(name).strip()
        match = re.match(r"^([A-Za-z]+[0-9]{4})", text)
        sources.append(match.group(1) if match else text.split("-", 1)[0])
    return np.asarray(sources, dtype=object)


def source_target_strata(targets, names):


    targets = np.asarray(targets, dtype=float)
    sources = publication_sources(names)
    counts = pd.Series(sources).value_counts()
    buckets = np.asarray(
        [source if counts[source] >= 2 * N_SPLITS else "Other" for source in sources],
        dtype=object,
    )
    labels = np.empty(len(targets), dtype=object)
    local_bins = {}
    for bucket in sorted(set(buckets)):
        positions = np.flatnonzero(buckets == bucket)
        requested_bins = min(NUM_STRATIFICATION_BINS, len(positions) // N_SPLITS)
        if requested_bins < 1:
            raise ValueError(f"Too few samples in source bucket {bucket}: {len(positions)}")
        while requested_bins >= 1:
            codes, edges = pd.qcut(
                targets[positions], q=requested_bins, labels=False,
                retbins=True, duplicates="drop",
            )
            codes = np.asarray(codes, dtype=int)
            _, bin_counts = np.unique(codes, return_counts=True)
            if len(bin_counts) and int(bin_counts.min()) >= N_SPLITS:
                break
            requested_bins -= 1
        if requested_bins < 1:
            raise ValueError(f"Cannot construct five-fold strata for source bucket {bucket}.")
        for position, code in zip(positions, codes):
            labels[position] = f"{bucket}::q{int(code)}"
        local_bins[bucket] = {
            "rows": int(len(positions)),
            "bins": int(len(np.unique(codes))),
            "edges": [float(value) for value in edges],
        }
    _, global_edges = regression_strata(targets)
    label_counts = pd.Series(labels).value_counts().sort_index()
    if int(label_counts.min()) < N_SPLITS:
        raise RuntimeError("Source-target strata are too small for five-fold CV.")
    audit = {
        "strategy": "publication_source_and_within_source_target_quantiles",
        "raw_source_counts": counts.sort_index().astype(int).to_dict(),
        "source_bucket_counts": pd.Series(buckets).value_counts().sort_index().astype(int).to_dict(),
        "local_target_bins": local_bins,
        "stratum_counts": label_counts.astype(int).to_dict(),
        "sources": sources.tolist(),
        "source_buckets": buckets.tolist(),
        "strata": labels.tolist(),
    }
    return labels, global_edges, audit


def _canonical_group_smiles(value, column, sample_name):

    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(
            f"Cannot canonicalize {column} for Co sample {sample_name!r}."
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def reaction_group_ids(data, sample_names):


    required = [
        column for column in REACTION_GROUP_COLUMNS[1:] if column not in data.columns
    ]
    if required:
        raise KeyError(f"Co transfer data is missing grouping columns: {required}")
    rows_by_name = {}
    for _, row in data.iterrows():
        rows_by_name.setdefault(str(row["Name"]), []).append(row)

    group_ids = []
    for sample_name in sample_names:
        matching_rows = rows_by_name.get(str(sample_name), [])
        if not matching_rows:
            raise KeyError(f"Cannot map graph sample {sample_name!r} back to the Co table.")
        row_group_ids = set()
        for row in matching_rows:
            catalyst = row.get("Cat", "Co")
            if pd.isna(catalyst) or not str(catalyst).strip():
                catalyst = "Co"
            components = [str(catalyst).strip()]
            components.extend(
                _canonical_group_smiles(row[column], column, sample_name)
                for column in REACTION_GROUP_COLUMNS[1:]
            )
            identity = "\x1f".join(components).encode("utf-8")
            row_group_ids.add(hashlib.sha256(identity).hexdigest()[:16])
        if len(row_group_ids) != 1:
            raise ValueError(
                f"Sample name {sample_name!r} maps to multiple reaction identities. "
                "Names must be unique or identify duplicate rows of the same reaction."
            )
        group_ids.append(row_group_ids.pop())
    return np.asarray(group_ids, dtype=object)


def make_cv_splits(strategy, strata, groups, seed):

    if strategy == "group_source_target":
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=seed
        )
        return list(
            splitter.split(np.zeros(len(strata)), strata, groups=groups)
        )
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(strata)), strata))


def take(dataset, indices):
    return [dataset[int(index)] for index in indices]


def set_requires_grad(model, trainable_module_names):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in trainable_module_names:
        module = model
        for component in module_name.split("."):
            module = getattr(module, component)
        for parameter in module.parameters():
            parameter.requires_grad = True


def stage2_trainable_modules(strategy):
    if strategy == "global_projection":
        return list(STAGE2_GLOBAL_PROJECTION_MODULES)
    if strategy == "full_encoders":
        return list(STAGE2_FULL_ENCODER_MODULES)
    raise ValueError(f"Unknown Stage 2 strategy: {strategy}")


def set_transfer_dropout(model, probability):

    probability = float(probability)
    if not 0.0 <= probability < 1.0:
        raise ValueError("transfer_dropout must be in [0, 1).")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = probability


def parameter_report(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(trainable), int(total)


def copy_state_to_cpu(model):
    return deepcopy(
        {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
    )


def save_transfer_history(stage1_history, stage2_history, output_dir):
    rows = []
    for stage_name, history in (
        ("stage1_head", stage1_history),
        ("stage2_finetune", stage2_history),
    ):
        if not history:
            continue
        for epoch in range(len(history["train_loss"])):
            rows.append(
                {
                    "stage": stage_name,
                    "epoch": epoch + 1,
                    "train_mse": float(history["train_loss"][epoch]),
                    "validation_r2": float(history["validation_r2"][epoch]),
                    "validation_rmse": float(
                        history["validation_rmse"][epoch]
                    ),
                    "validation_mae": float(history["validation_mae"][epoch]),
                    "learning_rate": float(history["learning_rate"][epoch]),
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "transfer_training_history.csv"), index=False
    )
    if stage1_history:
        plot_training_history(
            stage1_history,
            os.path.join(output_dir, "stage1_head_training_history.png"),
        )
    if stage2_history:
        plot_training_history(
            stage2_history,
            os.path.join(output_dir, "stage2_finetune_training_history.png"),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transfer the Ir Delta-G MCGNN checkpoint to Co using five-fold CV."
        )
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--xtb_root",
        default=DEFAULT_XTB_ROOT,
        help=(
            "Co xTB feature directory containing "
            "xtb_unique_molecule_features.csv and calculations/."
        ),
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stable_model_path",
        default=STABLE_MODEL_PATH,
        help="Stable checkpoint destination; override to keep ablations isolated.",
    )
    parser.add_argument("--cv_seed", type=int, default=10)
    parser.add_argument(
        "--cv_strategy",
        choices=("group_source_target", "source_target", "target_only"),
        default="group_source_target",
        help=(
            "group_source_target keeps each molecular reaction identity in one "
            "fold while balancing source/target strata; the other modes reproduce "
            "legacy row-wise splits."
        ),
    )
    parser.add_argument("--training_seed", type=int, default=666)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage2_epochs", type=int, default=150)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_lr", type=float, default=3e-5)
    parser.add_argument(
        "--stage2_strategy",
        choices=("global_projection", "full_encoders"),
        default="global_projection",
        help=(
            "global_projection adapts about 1.24M parameters and is the "
            "small-Co default; full_encoders reproduces the earlier ~8.35M "
            "parameter fine-tuning."
        ),
    )
    parser.add_argument("--transfer_dropout", type=float, default=0.20)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--scheduler_factor", type=float, default=0.50)
    parser.add_argument("--minimum_learning_rate", type=float, default=1e-8)
    parser.add_argument("--mem_fraction", type=float, default=0.7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    return parser.parse_args()


def run_transfer_for_fold(
    args,
    fold,
    train_indices,
    validation_indices,
    dataset,
    global_feature_dim,
    environment_global_feature_dim,
    device,
    source_state_dict,
    source_metadata,
    source_model_config,
    feature_settings,
    xtb_data_provenance,
    stratification_edges,
    reaction_groups,
):
    fold_output_dir = os.path.join(args.output_dir, f"fold_{int(fold)}")
    os.makedirs(fold_output_dir, exist_ok=True)
    setup_all_seeds(args.training_seed)
    train_data = take(dataset, train_indices)
    validation_data = take(dataset, validation_indices)
    print(
        f"\n===== Co Delta-G transfer fold {fold}/{N_SPLITS} =====\n"
        f"Train/Validation: {len(train_data)}/{len(validation_data)}"
    )

    model, model_config = build_regression_model_from_checkpoint(
        source_state_dict, source_metadata
    )
    if int(model_config["global_feature_dim"]) != int(global_feature_dim):
        raise ValueError(
            "Source checkpoint molecular global-feature dimension does not "
            "match the Co graphs."
        )
    if int(model_config["environment_global_feature_dim"]) != int(
        environment_global_feature_dim
    ):
        raise ValueError(
            "Source checkpoint environment global-feature dimension does not "
            "match the Co graphs."
        )
    model = model.to(device)
    model.load_state_dict(source_state_dict, strict=True)
    set_transfer_dropout(model, args.transfer_dropout)


    set_requires_grad(model, STAGE1_TRAINABLE_MODULES)
    stage1_trainable, total_parameters = parameter_report(model)
    print(
        f"Stage 1 trainable parameters: "
        f"{stage1_trainable:,}/{total_parameters:,}"
    )
    stage1_history = train_regressor(
        model,
        train_data,
        validation_data,
        device,
        num_epochs=args.stage1_epochs,
        batch_size=args.batch_size,
        learning_rate=args.stage1_lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        minimum_learning_rate=args.minimum_learning_rate,
        early_stopping_patience=args.patience,
        restore_best_weights=True,
    )
    stage1_state = copy_state_to_cpu(model)
    stage1_score = float(stage1_history["best_validation_r2"])
    final_stage = "stage1_head"

    stage2_history = None
    stage2_trainable = 0
    if not args.skip_stage2:


        stage2_modules = stage2_trainable_modules(args.stage2_strategy)
        set_requires_grad(model, stage2_modules)
        stage2_trainable, _ = parameter_report(model)
        print(
            f"Stage 2 strategy: {args.stage2_strategy}\n"
            f"Stage 2 trainable parameters: "
            f"{stage2_trainable:,}/{total_parameters:,}"
        )
        stage2_history = train_regressor(
            model,
            train_data,
            validation_data,
            device,
            num_epochs=args.stage2_epochs,
            batch_size=args.batch_size,
            learning_rate=args.stage2_lr,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            minimum_learning_rate=args.minimum_learning_rate,
            early_stopping_patience=args.patience,
            restore_best_weights=True,
        )
        stage2_score = float(stage2_history["best_validation_r2"])
        if stage2_score <= stage1_score:
            model.load_state_dict(stage1_state, strict=True)
            print(
                "Stage 2 did not improve validation R2; restored Stage 1 weights."
            )
        else:
            final_stage = f"stage2_{args.stage2_strategy}"

    save_transfer_history(
        stage1_history, stage2_history, fold_output_dir
    )
    train_metrics, _ = evaluate_and_save(
        model,
        train_data,
        device,
        os.path.join(fold_output_dir, "train_regression_results.xlsx"),
        f"Fold {fold} Train",
        batch_size=args.batch_size,
        target_column=TARGET_COLUMN,
        classification_dataset="Co",
    )
    validation_metrics, validation_table = evaluate_and_save(
        model,
        validation_data,
        device,
        os.path.join(fold_output_dir, "validation_regression_results.xlsx"),
        f"Fold {fold} Validation",
        batch_size=args.batch_size,
        target_column=TARGET_COLUMN,
        classification_dataset="Co",
    )
    validation_table.insert(0, "Dataset Position", validation_indices.astype(int))
    validation_table.insert(1, "Fold", int(fold))

    summary = {
        "fold": int(fold),
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "total_parameters": total_parameters,
        "stage1_trainable_parameters": stage1_trainable,
        "stage2_trainable_parameters": stage2_trainable,
        "stage2_strategy": args.stage2_strategy,
        "stage1_epochs_run": len(stage1_history["train_loss"]),
        "stage1_best_epoch": int(stage1_history["best_epoch"]),
        "stage1_best_validation_r2": stage1_score,
        "stage2_epochs_run": int(
            len(stage2_history["train_loss"]) if stage2_history else 0
        ),
        "stage2_best_epoch": int(
            stage2_history["best_epoch"] if stage2_history else 0
        ),
        "stage2_best_validation_r2": float(
            stage2_history["best_validation_r2"]
            if stage2_history
            else np.nan
        ),
        "final_stage": final_stage,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{
            f"validation_{key}": value
            for key, value in validation_metrics.items()
        },
    }
    metrics = {"train": train_metrics, "validation": validation_metrics}
    train_group_ids = sorted(set(reaction_groups[train_indices]))
    validation_group_ids = sorted(set(reaction_groups[validation_indices]))
    overlapping_group_ids = sorted(
        set(train_group_ids).intersection(validation_group_ids)
    )
    if args.cv_strategy == "group_source_target" and overlapping_group_ids:
        raise RuntimeError(
            "Grouped CV invariant failed: reaction groups occur in both the "
            f"training and validation split: {overlapping_group_ids}"
        )

    checkpoint = {
        "model_state_dict": copy_state_to_cpu(model),
        "metadata": {
            "task": "delta_g_regression_transfer_ir_to_co",
            "target_column": TARGET_COLUMN,
            "output_dimension": 1,
            "latent_output_dimension": 2,
            "head": PHYSICS_HEAD,
            "thermodynamic_relation": dict(THERMODYNAMIC_RELATION),
            "validation_method": {
                "group_source_target": (
                    "reaction_grouped_source_target_stratified_5_fold_cross_validation"
                ),
                "source_target": "source_target_stratified_5_fold_cross_validation",
                "target_only": "quantile_stratified_5_fold_cross_validation",
            }[args.cv_strategy],
            "stratification_edges": stratification_edges.tolist(),
            "global_feature_dim": int(global_feature_dim),
            "environment_global_feature_dim": int(
                environment_global_feature_dim
            ),
            "fingerprint_config": dict(
                source_metadata.get("fingerprint_config", {}) or {}
            ),
            "global_descriptor_config": dict(
                source_metadata.get("global_descriptor_config", {}) or {}
            ),
            "graph_ablation": {
                "disabled_branches": list(model_config.get("ablate_branches", ())),
                "definition": (
                    "Inherited from the Ir source checkpoint; reactant ablation "
                    "also removes the catalyst-context route derived from the "
                    "reactant graph."
                ),
            },
            "xtb_feature_config": dict(source_metadata["xtb_feature_config"]),


            "xtb_data_provenance": xtb_data_provenance,
            "reaction_site_index_features": REACTION_SITE_INDEX_FEATURE_LAYOUT,
            "node_feature_schema": list(NODE_FEATURE_NAMES),
            "edge_feature_schema": list(EDGE_FEATURE_NAMES),
            "model_config": model_config,
            "training_config": {
                "loss": "MSELoss",
                "optimizer": "AdamW",
                "batch_size": int(args.batch_size),
                "stage1_epochs": int(args.stage1_epochs),
                "stage2_epochs": int(0 if args.skip_stage2 else args.stage2_epochs),
                "stage1_learning_rate": float(args.stage1_lr),
                "stage2_learning_rate": float(args.stage2_lr),
                "transfer_dropout": float(args.transfer_dropout),
                "stage1_trainable_modules": list(STAGE1_TRAINABLE_MODULES),
                "stage2_strategy": args.stage2_strategy,
                "stage2_trainable_modules": stage2_trainable_modules(
                    args.stage2_strategy
                ),
                "weight_decay": float(args.weight_decay),
                "scheduler": "ReduceLROnPlateau",
                "scheduler_monitor": "validation_r2",
                "scheduler_patience": int(args.scheduler_patience),
                "scheduler_factor": float(args.scheduler_factor),
                "minimum_learning_rate": float(args.minimum_learning_rate),
                "early_stopping_patience": int(args.patience),
                "restore_best_validation_weights": True,
            },
            "source_checkpoint": os.path.abspath(args.checkpoint),
            "source_checkpoint_task": source_metadata.get("task"),
            "source_checkpoint_fold": source_metadata.get("cv_fold"),
            "source_checkpoint_best_epoch": source_metadata.get("best_epoch"),
            "source_model_config": source_model_config,
            "data_path": os.path.abspath(args.data),
            "default_catalyst": "Co",
            "n_splits": N_SPLITS,
            "cv_strategy": args.cv_strategy,
            "cv_seed": int(args.cv_seed),
            "training_seed": int(args.training_seed),
            "cv_fold": int(fold),
            "train_indices": [int(index) for index in train_indices],
            "validation_indices": [
                int(index) for index in validation_indices
            ],
            "reaction_grouping": {
                "columns": list(REACTION_GROUP_COLUMNS),
                "excludes": ["sol", TEMPERATURE_CELSIUS_COLUMN, TARGET_COLUMN],
                "train_group_ids": train_group_ids,
                "validation_group_ids": validation_group_ids,
                "overlap_group_ids": overlapping_group_ids,
            },
            "final_stage": final_stage,
            "feature_settings": feature_settings,
            "metrics": metrics,
        },
    }
    model_path = os.path.join(
        fold_output_dir, "co_xtb_metal_transfer_regression_checkpoint.pth"
    )
    torch.save(checkpoint, model_path)
    summary["model_path"] = os.path.abspath(model_path)
    return summary, validation_table


def regression_baseline_main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    stable_model_path = os.path.abspath(args.stable_model_path)
    os.makedirs(os.path.dirname(stable_model_path), exist_ok=True)
    device = (
        torch.device("cpu")
        if args.cpu
        else get_device(mem_fraction=args.mem_fraction)
    )

    source_state_dict, source_metadata, source_model_config = (
        read_regression_checkpoint(args.checkpoint)
    )
    validate_feature_schema(source_metadata)
    feature_settings = feature_settings_from_checkpoint(
        source_state_dict, source_metadata
    )
    data = load_table(args.data)
    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"Co transfer data is missing '{TARGET_COLUMN}'.")
    prepared_data_path = os.path.join(args.output_dir, "co_transfer_dataset.csv")
    data.to_csv(prepared_data_path, index=False, encoding="utf-8-sig")

    valid_indices = filter_valid_indices(data, default_catalyst="Co")
    xtb_store = (
        get_xtb_store(args.xtb_root)
        if feature_settings["use_xtb_features"]
        else None
    )
    graph_storage_device = torch.device("cpu")
    dataset, global_feature_dim = build_dataset(
        data,
        valid_indices,
        graph_storage_device,
        target_column=TARGET_COLUMN,
        default_catalyst="Co",
        xtb_store=xtb_store,
        **feature_settings,
    )
    xtb_data_provenance = xtb_store.report() if xtb_store is not None else None
    if xtb_store is not None:
        with open(os.path.join(args.output_dir, "xtb_feature_coverage.json"), "w", encoding="utf-8") as stream:
            json.dump(xtb_data_provenance, stream, ensure_ascii=False, indent=2)
    if len(dataset) < N_SPLITS:
        raise RuntimeError(f"Too few valid Co transfer samples: {len(dataset)}")
    environment_global_feature_dim = int(
        dataset[0][1].ndata["global"].shape[1]
    )
    if int(source_model_config["global_feature_dim"]) != int(global_feature_dim):
        raise ValueError(
            "Checkpoint and Co molecular global-feature dimensions differ: "
            f"{source_model_config['global_feature_dim']} vs {global_feature_dim}."
        )
    if int(source_model_config["environment_global_feature_dim"]) != int(
        environment_global_feature_dim
    ):
        raise ValueError(
            "Checkpoint and Co environment global-feature dimensions differ: "
            f"{source_model_config['environment_global_feature_dim']} vs "
            f"{environment_global_feature_dim}."
        )

    targets = np.asarray([float(sample[-1]) for sample in dataset], dtype=float)
    sample_names = [sample[0] for sample in dataset]
    if args.cv_strategy in {"group_source_target", "source_target"}:
        strata, stratification_edges, split_audit = source_target_strata(
            targets, sample_names
        )
    else:
        strata, stratification_edges = regression_strata(targets)
        sources = publication_sources(sample_names)
        split_audit = {
            "strategy": "target_quantiles_only",
            "raw_source_counts": pd.Series(sources).value_counts().sort_index().astype(int).to_dict(),
            "source_bucket_counts": pd.Series(sources).value_counts().sort_index().astype(int).to_dict(),
            "sources": sources.tolist(), "source_buckets": sources.tolist(),
            "strata": [str(value) for value in strata],
        }
    reaction_groups = reaction_group_ids(data, sample_names)
    group_counts = pd.Series(reaction_groups).value_counts()
    split_audit.update(
        {
            "reaction_group_columns": list(REACTION_GROUP_COLUMNS),
            "reaction_group_excludes": [
                "sol", TEMPERATURE_CELSIUS_COLUMN, TARGET_COLUMN
            ],
            "reaction_group_count": int(group_counts.size),
            "duplicate_reaction_group_count": int((group_counts > 1).sum()),
            "rows_in_duplicate_reaction_groups": int(
                group_counts[group_counts > 1].sum()
            ),
        }
    )
    print(f"Source checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Co data: {os.path.abspath(args.data)}")
    if xtb_store is not None:
        print(f"Co xTB source: {xtb_store.root}")
        print(f"Co xTB coverage: {xtb_data_provenance['coverage']}")
    print(f"Valid Co samples: {len(dataset)}")
    print(
        "Inherited source architecture: "
        f"reactant {source_model_config['reactant_hidden_feats']}/"
        f"{source_model_config['reactant_out_feats']}, product "
        f"{source_model_config['product_hidden_feats']}/"
        f"{source_model_config['product_out_feats']}, "
        f"FC {source_model_config['fc_hidden_feats']}"
    )
    print(
        f"Inherited fingerprint width: {feature_settings['fp_nbits']} bits; "
        f"3D descriptors={feature_settings['include_3d_descriptors']}"
    )

    all_summaries = []
    oof_tables = []
    cv_splits = make_cv_splits(
        args.cv_strategy, strata, reaction_groups, args.cv_seed
    )
    fold_assignments = np.zeros(len(dataset), dtype=int)
    for fold, (_, validation_indices) in enumerate(cv_splits, start=1):
        fold_assignments[validation_indices] = fold
    group_size_by_row = [int(group_counts[group]) for group in reaction_groups]
    split_rows = pd.DataFrame({
        "Dataset Position": np.arange(len(dataset)), "Sample Name": sample_names,
        "Reaction Group ID": reaction_groups,
        "Reaction Group Size": group_size_by_row,
        "Fold": fold_assignments, "Source": split_audit["sources"],
        "Source Bucket": split_audit["source_buckets"],
        "Stratum": split_audit["strata"], TARGET_COLUMN: targets,
    })
    group_fold_counts = split_rows.groupby("Reaction Group ID")["Fold"].nunique()
    groups_split_across_folds = group_fold_counts[group_fold_counts > 1].index.tolist()
    split_audit["groups_split_across_folds"] = groups_split_across_folds
    if args.cv_strategy == "group_source_target" and groups_split_across_folds:
        raise RuntimeError(
            "Grouped CV placed reaction identities in multiple folds: "
            f"{groups_split_across_folds}"
        )
    split_rows.to_csv(
        os.path.join(args.output_dir, "co_cv_split_audit.csv"),
        index=False, encoding="utf-8-sig",
    )
    with open(os.path.join(args.output_dir, "co_cv_split_audit.json"), "w", encoding="utf-8") as stream:
        json.dump({key: value for key, value in split_audit.items()
                   if key not in {"sources", "source_buckets", "strata"}},
                  stream, ensure_ascii=False, indent=2)
    for fold, (train_indices, validation_indices) in enumerate(cv_splits, start=1):
        summary, validation_table = run_transfer_for_fold(
            args,
            fold,
            train_indices,
            validation_indices,
            dataset,
            global_feature_dim,
            environment_global_feature_dim,
            device,
            source_state_dict,
            source_metadata,
            source_model_config,
            feature_settings,
            xtb_data_provenance,
            stratification_edges,
            reaction_groups,
        )
        all_summaries.append(summary)
        oof_tables.append(validation_table)

    summary_df = pd.DataFrame(all_summaries)
    oof_df = pd.concat(oof_tables, ignore_index=True).sort_values(
        "Dataset Position"
    )
    if len(oof_df) != len(dataset) or oof_df["Dataset Position"].nunique() != len(
        dataset
    ):
        raise RuntimeError(
            "OOF coverage error: every valid Co sample must appear exactly once."
        )
    true_column = f"True {TARGET_COLUMN}"
    predicted_column = f"Predicted {TARGET_COLUMN}"
    oof_metrics = regression_metrics(
        oof_df[true_column], oof_df[predicted_column]
    )
    from src.regression_classification import export_classification
    oof_df = export_classification(oof_df, args.output_dir, "Co", stem="oof_classification",
                                   role="Ir-to-Co transfer grouped OOF: regression to classes")
    aggregate_rows = []
    for metric_name in ("r2", "rmse", "mae"):
        validation_column = f"validation_{metric_name}"
        aggregate_rows.append(
            {
                "metric": metric_name,
                "fold_mean": float(summary_df[validation_column].mean()),
                "fold_std": float(summary_df[validation_column].std(ddof=1)),
                "oof_value": float(oof_metrics[metric_name]),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_path = os.path.join(
        args.output_dir, "transfer_five_fold_cross_validation_summary.xlsx"
    )
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Fold Summary", index=False)
        aggregate_df.to_excel(writer, sheet_name="Mean Std and OOF", index=False)
        oof_df.to_excel(writer, sheet_name="OOF Predictions", index=False)
        pd.DataFrame(
            {
                "lower_edge": stratification_edges[:-1],
                "upper_edge": stratification_edges[1:],
            }
        ).to_excel(writer, sheet_name="Target Strata", index=False)
        split_rows.to_excel(writer, sheet_name="CV Split Audit", index=False)
    oof_df.to_csv(
        os.path.join(args.output_dir, "transfer_five_fold_oof_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_regression_scatter(
        oof_df[true_column],
        oof_df[predicted_column],
        os.path.join(args.output_dir, "transfer_five_fold_oof_scatter.png"),
        "Co Transfer Five-fold OOF",
        target_label=TARGET_COLUMN,
    )

    best_row = summary_df.sort_values(
        ["validation_r2", "validation_rmse"], ascending=[False, True]
    ).iloc[0]
    pd.DataFrame([best_row.to_dict()]).to_excel(
        os.path.join(args.output_dir, "best_transfer_regression_fold.xlsx"),
        index=False,
    )
    best_output_model = os.path.join(
        args.output_dir, "best_co_xtb_metal_transfer_regression_checkpoint.pth"
    )
    shutil.copyfile(best_row["model_path"], best_output_model)
    shutil.copyfile(best_output_model, stable_model_path)
    print(f"\nFive-fold mean/std and OOF metrics:\n{aggregate_df}")
    print(f"Best transfer fold by validation R2: {int(best_row['fold'])}")
    print(f"Summary: {summary_path}")
    print(f"Best transfer checkpoint: {best_output_model}")
    print(f"Stable checkpoint copied to: {stable_model_path}")


def main():

    if "--regression-baseline" in sys.argv:
        sys.argv.remove("--regression-baseline")
        regression_baseline_main()
    else:
        sys.path.insert(0, PROJECT_DIR)
        from main import main as fixed_main
        fixed_main(["co-joint", *sys.argv[1:]])


if __name__ == "__main__":
    main()
