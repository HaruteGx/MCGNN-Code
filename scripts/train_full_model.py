import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import rdBase
from sklearn.model_selection import StratifiedKFold


rdBase.DisableLog("rdApp.warning")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

from src.config import get_device, setup_seed
from src.data_splits import ensure_random_ir_split, checkpoint_split_config
from src.ir_protocol import TEST_SPLIT_SEED, TEST_FRACTION, ir_run_paths
from src.holdout_eval import evaluate_frozen_holdout
from src.dataset import (
    PRODUCT_SITE_INDEX_COLUMN,
    REACTION_SITE_INDEX_FEATURE_LAYOUT,
    SUBSTRATE_SITE_INDEX_COLUMN,
    build_dataset,
    filter_valid_indices,
    parse_reaction_site_indices,
)
from src.descriptors import (
    environment_global_feature_dimension,
    global_descriptor_config,
)
from src.features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    REACTION_SITE_NODE_FEATURE_INDEX,
)
from src.models import ReactionEnergyPredictionModel
from src.regression_classification import export_classification
from src.xtb_features import get_xtb_store, xtb_feature_config
from src.train_eval import (
    evaluate_and_save,
    plot_training_history,
    regression_metrics,
    save_regression_scatter,
    train_regressor,
)


PROTOCOL_PATHS = ir_run_paths(PROJECT_DIR)
DATA_PATH = str(PROTOCOL_PATHS["full"])
DEVELOPMENT_DATA_PATH = str(PROTOCOL_PATHS["development"])
TEST_DATA_PATH = str(PROTOCOL_PATHS["test"])
SPLIT_MANIFEST_PATH = str(PROTOCOL_PATHS["manifest"])
OUTPUT_DIR = str(PROTOCOL_PATHS["output"])
STABLE_MODEL_PATH = str(PROTOCOL_PATHS["stable_checkpoint"])
TARGET_COLUMN = "G (kcal/mol)"
DEFAULT_CATALYST = "Ir"
FINGERPRINT_NBITS = 1024
INCLUDE_3D_DESCRIPTORS = True
DESCRIPTOR_NUM_CONFORMERS = 3
DESCRIPTOR_CONFORMER_SEED = 20260822
OPTIMIZE_CONFORMERS = True
USE_XTB_FEATURES = True


N_SPLITS = 5
NUM_STRATIFICATION_BINS = 10
CV_SEED = 10
TRAINING_SEED = 63


NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.40
MINIMUM_LEARNING_RATE = 1e-8
EARLY_STOPPING_PATIENCE = 15
RESTORE_BEST_WEIGHTS = True
MODEL_CONFIG = {
    "dropout": 0.20,
    "reactant_hidden_feats": 512,
    "reactant_out_feats": 256,
    "product_hidden_feats": 512,
    "product_out_feats": 256,
    "fc_hidden_feats": 128,
    "catalyst_context_dim": 64,
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
    edges = np.asarray(edges, dtype=float)
    labels, counts = np.unique(strata, return_counts=True)
    if len(labels) < 2:
        raise ValueError("The regression target has insufficient variation.")
    if counts.min() < N_SPLITS:
        raise ValueError(
            f"Five-fold stratification needs at least {N_SPLITS} samples in every "
            f"target bin; counts are {dict(zip(labels.tolist(), counts.tolist()))}."
        )
    return strata, edges, dict(zip(labels.tolist(), counts.tolist()))


def take(dataset, indices):
    return [dataset[int(index)] for index in indices]


def build_reaction_site_feature_audit(data, valid_source_rows, dataset):

    if len(valid_source_rows) != len(dataset):
        raise RuntimeError(
            "Dataset/source-row alignment was lost during graph construction."
        )
    rows = []
    for dataset_position, (source_row, sample) in enumerate(
        zip(valid_source_rows, dataset)
    ):
        source_row = int(source_row)
        row = data.iloc[source_row]
        substrate_indices = parse_reaction_site_indices(
            row[SUBSTRATE_SITE_INDEX_COLUMN],
            SUBSTRATE_SITE_INDEX_COLUMN,
            source_row,
        )
        product_indices = parse_reaction_site_indices(
            row[PRODUCT_SITE_INDEX_COLUMN],
            PRODUCT_SITE_INDEX_COLUMN,
            source_row,
        )
        reactant_count = int(
            (
                sample[2].ndata["node"][:, REACTION_SITE_NODE_FEATURE_INDEX]
                == 1
            ).sum().item()
        )
        product_count = int(
            (
                sample[3].ndata["node"][:, REACTION_SITE_NODE_FEATURE_INDEX]
                == 1
            ).sum().item()
        )
        if reactant_count != len(substrate_indices) or product_count != len(
            product_indices
        ):
            raise RuntimeError(
                f"Reaction-site feature mapping mismatch at source row {source_row}."
            )
        rows.append(
            {
                "Dataset Position": dataset_position,
                "Source Row Index": source_row,
                "Sample Name": sample[0],
                SUBSTRATE_SITE_INDEX_COLUMN: str(
                    [index + 1 for index in substrate_indices]
                ),
                PRODUCT_SITE_INDEX_COLUMN: str(
                    [index + 1 for index in product_indices]
                ),
                "Reactant Marked Node Count": reactant_count,
                "Product Marked Node Count": product_count,
            }
        )
    return pd.DataFrame(rows)


def make_checkpoint(
    model,
    global_feature_dim,
    environment_global_dim,
    fold,
    train_indices,
    validation_indices,
    valid_source_rows,
    metrics,
    best_epoch,
    stratification_edges,
    split_manifest,
):
    descriptor_metadata = global_descriptor_config(
        include_3d=INCLUDE_3D_DESCRIPTORS,
        max_conformers=DESCRIPTOR_NUM_CONFORMERS,
        conformer_seed=DESCRIPTOR_CONFORMER_SEED,
        optimize_conformers=OPTIMIZE_CONFORMERS,
    )
    return {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": {
            "task": "reaction_energy_regression",
            "target_column": TARGET_COLUMN,
            "output_dimension": 1,
            "latent_output_dimension": 2,
            "head": "enthalpy_entropy_physics",
            "thermodynamic_relation": {
                "temperature_source_column": "T / C",
                "temperature_model_unit": "K",
                "delta_h_unit": "kcal/mol",
                "delta_s_unit": "cal/mol/K",
                "delta_g_unit": "kcal/mol",
                "formula": "Delta G = Delta H - T * Delta S / 1000",
                "loss": "MSE(predicted Delta G, measured Delta G)",
            },
            "validation_method": "quantile_stratified_5_fold_cross_validation",
            "stratification_edges": [float(value) for value in stratification_edges],
            "global_feature_dim": int(global_feature_dim),
            "environment_global_feature_dim": int(environment_global_dim),
            "fingerprint_config": {
                "radius": 2,
                "n_bits_per_fingerprint": FINGERPRINT_NBITS,
                "fingerprints_per_molecular_graph": 2,
                "environment_fingerprint_nbits": 256,
            },
            "global_descriptor_config": descriptor_metadata,
            "xtb_feature_config": xtb_feature_config(USE_XTB_FEATURES),
            "xtb_data_provenance": get_xtb_store().report() if USE_XTB_FEATURES else None,
            "reaction_site_index_features": REACTION_SITE_INDEX_FEATURE_LAYOUT,
            "node_feature_schema": list(NODE_FEATURE_NAMES),
            "edge_feature_schema": list(EDGE_FEATURE_NAMES),
            "model_config": {
                "global_feature_dim": int(global_feature_dim),
                "environment_global_feature_dim": int(environment_global_dim),
                **MODEL_CONFIG,
            },
            "graph_ablation": {
                "disabled_branches": list(MODEL_CONFIG.get("ablate_branches", ())),
                "definition": (
                    "The selected branch embedding is zeroed before fusion; "
                    "reactant ablation also removes the catalyst-context route "
                    "derived from the reactant graph."
                ),
            },
            "training_config": {
                "loss": "MSELoss",
                "optimizer": "AdamW",
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": "ReduceLROnPlateau",
                "scheduler_monitor": "validation_r2",
                "scheduler_patience": SCHEDULER_PATIENCE,
                "scheduler_factor": SCHEDULER_FACTOR,
                "minimum_learning_rate": MINIMUM_LEARNING_RATE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "restore_best_validation_weights": RESTORE_BEST_WEIGHTS,
            },
            "default_catalyst": DEFAULT_CATALYST,
            "data_path": os.path.abspath(DATA_PATH),
            "development_data_path": os.path.abspath(DEVELOPMENT_DATA_PATH),
            "holdout_split": checkpoint_split_config(split_manifest),
            "n_splits": N_SPLITS,
            "cv_seed": CV_SEED,
            "training_seed": TRAINING_SEED,
            "cv_fold": int(fold),
            "best_epoch": int(best_epoch),
            "train_indices": [int(index) for index in train_indices],
            "validation_indices": [int(index) for index in validation_indices],
            "valid_source_rows": [int(index) for index in valid_source_rows],
            "metrics": metrics,
        },
    }


def parameter_table(
    global_feature_dim,
    environment_global_dim,
    parameter_count,
    stratum_counts,
    edges,
):
    rows = [
        {"group": "task", "parameter": "target_column", "value": TARGET_COLUMN},
        {
            "group": "task",
            "parameter": "architecture",
            "value": "xTB graph encoder -> Delta H/Delta S -> Delta G = Delta H - T*Delta S/1000; Delta-G MSE",
        },
        {
            "group": "features",
            "parameter": "node_feature_schema",
            "value": json.dumps(list(NODE_FEATURE_NAMES)),
        },
        {
            "group": "features",
            "parameter": "edge_feature_schema",
            "value": json.dumps(list(EDGE_FEATURE_NAMES)),
        },
    ]
    parameters = {
        "fingerprint_nbits": FINGERPRINT_NBITS,
        "include_3d_descriptors": INCLUDE_3D_DESCRIPTORS,
        "descriptor_num_conformers": DESCRIPTOR_NUM_CONFORMERS,
        "use_xtb_features": USE_XTB_FEATURES,
        "reactant_product_global_feature_dim": global_feature_dim,
        "environment_global_feature_dim": environment_global_dim,
        "trainable_parameters": parameter_count,
        **MODEL_CONFIG,
        "loss": "MSELoss",
        "optimizer": "AdamW",
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }
    rows.extend(
        {"group": "model_training", "parameter": key, "value": value}
        for key, value in parameters.items()
    )
    rows.extend(
        [
            {
                "group": "cross_validation",
                "parameter": "stratification_edges",
                "value": json.dumps([float(value) for value in edges]),
            },
            {
                "group": "cross_validation",
                "parameter": "stratum_counts",
                "value": json.dumps(stratum_counts, sort_keys=True),
            },
        ]
    )
    return pd.DataFrame(rows)


def main(argv=None):
    global OUTPUT_DIR, STABLE_MODEL_PATH, MODEL_CONFIG, NUM_EPOCHS
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Ir Delta-H/Delta-S graph regressor."
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--stable-model-path",
        default=STABLE_MODEL_PATH,
        help="Checkpoint destination; defaults to the configured stable model path.",
    )
    ablation_group = parser.add_mutually_exclusive_group()
    ablation_group.add_argument(
        "--ablate-branch",
        choices=("none", "environment", "reactant", "product"),
        default="none",
        help="Zero one graph branch before fusion for a controlled ablation.",
    )
    ablation_group.add_argument(
        "--ablate-branches",
        nargs="+",
        choices=ReactionEnergyPredictionModel.ABLATABLE_GRAPH_BRANCHES,
        help="Zero multiple graph branches before fusion; reactant includes catalyst context.",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv if argv is not None else [])
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    STABLE_MODEL_PATH = os.path.abspath(args.stable_model_path)
    MODEL_CONFIG = dict(MODEL_CONFIG)
    MODEL_CONFIG["ablate_branches"] = (
        sorted(set(args.ablate_branches)) if args.ablate_branches
        else ([] if args.ablate_branch == "none" else [args.ablate_branch])
    )
    NUM_EPOCHS = int(args.epochs)
    if NUM_EPOCHS < 1:
        raise ValueError("--epochs must be a positive integer.")
    setup_seed(TRAINING_SEED)
    device = torch.device("cpu") if args.cpu else get_device(mem_fraction=0.7)
    graph_storage_device = torch.device("cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STABLE_MODEL_PATH), exist_ok=True)


    data, test_frame, split_manifest = ensure_random_ir_split(
        DATA_PATH, DEVELOPMENT_DATA_PATH, TEST_DATA_PATH, SPLIT_MANIFEST_PATH,
        seed=TEST_SPLIT_SEED, test_fraction=TEST_FRACTION,
    )
    with open(os.path.join(OUTPUT_DIR, "ir_holdout_split.json"), "w", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, ensure_ascii=False, indent=2)
    xtb_store = get_xtb_store() if USE_XTB_FEATURES else None
    valid_indices = filter_valid_indices(data, default_catalyst=DEFAULT_CATALYST)
    dataset, global_feature_dim = build_dataset(
        data,
        valid_indices,
        graph_storage_device,
        fp_nbits=FINGERPRINT_NBITS,
        target_column=TARGET_COLUMN,
        default_catalyst=DEFAULT_CATALYST,
        include_3d_descriptors=INCLUDE_3D_DESCRIPTORS,
        descriptor_num_conformers=DESCRIPTOR_NUM_CONFORMERS,
        descriptor_conformer_seed=DESCRIPTOR_CONFORMER_SEED,
        optimize_conformers=OPTIMIZE_CONFORMERS,
        use_xtb_features=USE_XTB_FEATURES,
        xtb_store=xtb_store,
    )
    if not dataset:
        raise RuntimeError("No valid regression samples were built.")
    if xtb_store is not None:
        with open(os.path.join(OUTPUT_DIR, "xtb_feature_coverage.json"), "w", encoding="utf-8") as stream:
            json.dump(xtb_store.report(), stream, ensure_ascii=False, indent=2)
    if len(dataset) != len(valid_indices):
        raise RuntimeError(
            "Some rows passed validation but failed graph construction; source-row "
            "alignment cannot be guaranteed."
        )

    environment_global_dim = environment_global_feature_dimension(
        256, include_3d=INCLUDE_3D_DESCRIPTORS
    )
    actual_environment_dim = int(dataset[0][1].ndata["global"].shape[1])
    actual_reactant_dim = int(dataset[0][2].ndata["global"].shape[1])
    actual_product_dim = int(dataset[0][3].ndata["global"].shape[1])
    if (
        actual_environment_dim != environment_global_dim
        or actual_reactant_dim != global_feature_dim
        or actual_product_dim != global_feature_dim
    ):
        raise RuntimeError("Constructed graph global dimensions are inconsistent.")

    audit = build_reaction_site_feature_audit(data, valid_indices, dataset)
    audit.insert(2, "Full Source Row Index", [
        split_manifest["development_source_row_indices_zero_based"][i] for i in valid_indices])
    audit_path = os.path.join(OUTPUT_DIR, "reaction_site_index_feature_audit.csv")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    targets = np.asarray([float(sample[-1]) for sample in dataset], dtype=float)
    strata, stratum_edges, stratum_counts = regression_strata(targets)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED
    )

    example_model = ReactionEnergyPredictionModel(
        global_feature_dim=global_feature_dim,
        environment_global_feature_dim=environment_global_dim,
        **MODEL_CONFIG,
    )
    parameter_count = sum(
        parameter.numel() for parameter in example_model.parameters()
    )
    del example_model

    print(f"Valid samples: {len(dataset)}")
    duplicate_inputs = split_manifest["audit"]["overlapping_model_input_test_rows"]
    print(
        f"Target: {TARGET_COLUMN}; range={targets.min():.4f} to "
        f"{targets.max():.4f}; mean={targets.mean():.4f}"
    )
    print(
        f"Node/edge dimensions: {len(NODE_FEATURE_NAMES)}/{len(EDGE_FEATURE_NAMES)}"
    )
    print(
        f"Environment/reactant/product global dimensions: "
        f"{environment_global_dim}/{global_feature_dim}/{global_feature_dim}"
    )
    print(f"Reaction-site audit: {audit_path}")

    fold_rows = []
    oof_frames = []
    best_score = (-float("inf"), -float("inf"))
    best_model_path = None

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(dataset)), strata), start=1
    ):
        setup_seed(TRAINING_SEED)
        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        train_data = take(dataset, train_indices)
        validation_data = take(dataset, validation_indices)
        print(
            f"\n===== Ir {TARGET_COLUMN} regression fold {fold}/{N_SPLITS} =====\n"
            f"Train/Validation: {len(train_data)}/{len(validation_data)}"
        )

        model = ReactionEnergyPredictionModel(
            global_feature_dim=global_feature_dim,
            environment_global_feature_dim=environment_global_dim,
            **MODEL_CONFIG,
        ).to(device)
        history = train_regressor(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            device=device,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            scheduler_patience=SCHEDULER_PATIENCE,
            scheduler_factor=SCHEDULER_FACTOR,
            minimum_learning_rate=MINIMUM_LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=RESTORE_BEST_WEIGHTS,
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
            os.path.join(fold_dir, "train_regression_results.xlsx"),
            f"Fold {fold} Train",
            batch_size=BATCH_SIZE,
            target_column=TARGET_COLUMN,
        )
        validation_metrics, validation_results = evaluate_and_save(
            model,
            validation_data,
            device,
            os.path.join(fold_dir, "validation_regression_results.xlsx"),
            f"Fold {fold} Validation",
            batch_size=BATCH_SIZE,
            target_column=TARGET_COLUMN,
        )

        oof = validation_results.copy()
        oof.insert(0, "Dataset Position", validation_indices.astype(int))
        oof.insert(
            1,
            "Source Row Index",
            [int(valid_indices[index]) for index in validation_indices],
        )
        oof.insert(2, "Fold", fold)
        oof_frames.append(oof)

        fold_row = {
            "fold": fold,
            "train_size": len(train_data),
            "validation_size": len(validation_data),
            "epochs_run": len(history["train_loss"]),
            "best_epoch": int(history["best_epoch"]),
            "best_epoch_validation_r2": float(history["best_validation_r2"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        fold_rows.append(fold_row)

        checkpoint = make_checkpoint(
            model,
            global_feature_dim,
            environment_global_dim,
            fold,
            train_indices,
            validation_indices,
            valid_indices,
            {"train": train_metrics, "validation": validation_metrics},
            history["best_epoch"],
            stratum_edges,
            split_manifest,
        )
        fold_model_path = os.path.join(
            fold_dir, "ir_xtb_metal_regression_checkpoint.pth"
        )
        torch.save(checkpoint, fold_model_path)
        score = (validation_metrics["r2"], -validation_metrics["rmse"])
        if score > best_score:
            best_score = score
            best_model_path = fold_model_path
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True).sort_values("Dataset Position")
    if len(oof_df) != len(dataset) or oof_df["Dataset Position"].nunique() != len(
        dataset
    ):
        raise RuntimeError("Every valid sample must appear in OOF exactly once.")
    true_column = f"True {TARGET_COLUMN}"
    predicted_column = f"Predicted {TARGET_COLUMN}"
    oof_metrics = regression_metrics(oof_df[true_column], oof_df[predicted_column])
    oof_df = export_classification(oof_df, OUTPUT_DIR, "Ir", stem="oof_classification",
                                   role="Ir development OOF: regression to classes")

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
    parameters_df = parameter_table(
        global_feature_dim,
        environment_global_dim,
        parameter_count,
        stratum_counts,
        stratum_edges,
    )
    summary_path = os.path.join(OUTPUT_DIR, "five_fold_cross_validation_summary.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name="Fold Metrics", index=False)
        aggregate_df.to_excel(writer, sheet_name="Mean Std and OOF", index=False)
        oof_df.to_excel(writer, sheet_name="OOF Predictions", index=False)
        parameters_df.to_excel(writer, sheet_name="Parameters", index=False)
    oof_df.to_csv(
        os.path.join(OUTPUT_DIR, "five_fold_oof_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_regression_scatter(
        oof_df[true_column],
        oof_df[predicted_column],
        os.path.join(OUTPUT_DIR, "five_fold_oof_scatter.png"),
        f"Five-fold OOF {TARGET_COLUMN} regression",
        target_label=TARGET_COLUMN,
    )

    if best_model_path is None:
        raise RuntimeError("No cross-validation checkpoint was produced.")
    best_output_model = os.path.join(
        OUTPUT_DIR, "best_ir_xtb_metal_regression_checkpoint.pth"
    )
    shutil.copyfile(best_model_path, best_output_model)
    shutil.copyfile(best_output_model, STABLE_MODEL_PATH)
    best_fold = int(
        fold_df.sort_values(
            ["validation_r2", "validation_rmse"], ascending=[False, True]
        ).iloc[0]["fold"]
    )


    test_report, test_results = evaluate_frozen_holdout(
        best_output_model, test_frame, split_manifest, device,
        os.path.join(OUTPUT_DIR, "test"), batch_size=BATCH_SIZE,
    )
    with pd.ExcelWriter(summary_path, engine="openpyxl", mode="a") as writer:
        pd.DataFrame([{
            "test_size": len(test_frame), "selected_cv_fold": best_fold,
            **test_report["metrics"],
        }]).to_excel(writer, sheet_name="Independent Test Metrics", index=False)
        test_results.to_excel(writer, sheet_name="Independent Test Predictions", index=False)
    print(f"\nFive-fold mean/std and OOF metrics:\n{aggregate_df}")
    print(f"Best fold by validation R2: {best_fold}")
    print(f"Summary: {summary_path}")
    print(f"Best checkpoint: {best_output_model}")
    print(f"Stable checkpoint copied to: {STABLE_MODEL_PATH}")
    print(f"Independent test results: {os.path.join(OUTPUT_DIR, 'test')}")


if __name__ == "__main__":
    main(sys.argv[1:])
