import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Any, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import dgl
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from . import graph_builders as gb
from .dataset import (
    PRODUCT_SITE_INDEX_COLUMN,
    SUBSTRATE_SITE_INDEX_COLUMN,
    TEMPERATURE_CELSIUS_COLUMN,
    temperature_kelvin_from_celsius,
)
from .mcencoding import build_global_features, encode_catalyst_feature, encode_ligand_fp
from .models import build_regression_model_from_checkpoint
from .xtb_features import get_xtb_store, reaction_xtb_vector
from .regression_classification import attach_classification, export_classification


REQUIRED_COLUMNS = (
    "Name",
    "sol",
    "sub-type",
    "L-type",
    "B-type",
    "P",
    TEMPERATURE_CELSIUS_COLUMN,
)


def _is_missing(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _parse_index_list(value: Any) -> List[int]:
    if _is_missing(value) or str(value).strip() == "":
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            parsed = [part.strip() for part in str(value).strip("[](){}").split(",")]
    if isinstance(parsed, (int, float, np.integer, np.floating)):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        return []
    return [int(float(item)) for item in parsed if str(item).strip()]


def _to_zero_based(indices, already_zero_based):
    return [int(index) if already_zero_based else int(index) - 1 for index in indices]


def _check_required_columns(dataframe, default_catalyst=None):
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if "Cat" not in dataframe.columns and default_catalyst is None:
        missing.append("Cat")
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")


def _fingerprint_nbits_from_model(model):
    saved_nbits = getattr(model, "fingerprint_nbits", None)
    if saved_nbits is None:
        raise ValueError(
            "Checkpoint metadata has no fingerprint width. Retrain with the current "
            "Regress Test training script."
        )
    return int(saved_nbits)


def _descriptor_settings_from_model(model):
    config = dict(getattr(model, "global_descriptor_config", {}) or {})
    return {
        "include_3d_descriptors": bool(config.get("include_3d", False)),
        "descriptor_num_conformers": int(config.get("max_conformers", 3)),
        "descriptor_conformer_seed": int(config.get("conformer_seed", 20260822)),
        "optimize_conformers": bool(config.get("optimize_conformers", True)),
    }


def _build_graphs_for_row(
    row,
    device,
    fingerprint_nbits,
    descriptor_settings,
    allow_missing_indices=False,
    indices_are_zero_based=False,
    default_catalyst=None,
    xtb_store=None,
):
    name = str(row["Name"])
    solvent = row["sol"]
    substrate = row["sub-type"]
    ligand = row["L-type"]
    bpin = row["B-type"]
    product = row["P"]
    catalyst = row.get("Cat", default_catalyst)
    if _is_missing(catalyst) or str(catalyst).strip() == "":
        catalyst = default_catalyst
    for column, value in (
        ("sol", solvent),
        ("sub-type", substrate),
        ("L-type", ligand),
        ("B-type", bpin),
        ("P", product),
        ("Cat", catalyst),
    ):
        if _is_missing(value) or str(value).strip() == "":
            return None, f"Missing value in {column}"

    catalyst_feature = encode_catalyst_feature(str(catalyst))
    if catalyst_feature is None:
        return None, f"Unsupported catalyst: {catalyst}"
    substrate_indices = _parse_index_list(
        row.get(SUBSTRATE_SITE_INDEX_COLUMN, None)
    )
    product_indices = _parse_index_list(row.get(PRODUCT_SITE_INDEX_COLUMN, None))
    if (not substrate_indices or not product_indices) and not allow_missing_indices:
        return None, (
            f"Missing {SUBSTRATE_SITE_INDEX_COLUMN} or "
            f"{PRODUCT_SITE_INDEX_COLUMN}"
        )
    substrate_indices = _to_zero_based(
        substrate_indices, indices_are_zero_based
    )
    product_indices = _to_zero_based(product_indices, indices_are_zero_based)

    try:
        environment_graph = gb.create_environment_graph(
            str(solvent), device=device, xtb_store=xtb_store, **descriptor_settings
        )
    except Exception as error:
        return None, f"Environment graph error: {error}"
    if environment_graph is None:
        return None, "Environment graph construction failed"

    ligand_fingerprint = encode_ligand_fp(
        str(ligand), nBits=fingerprint_nbits
    )
    if ligand_fingerprint is None:
        return None, "Ligand fingerprint construction failed"
    combined = gb.combine_smiles([str(substrate), str(bpin), str(ligand)])
    if combined is None:
        return None, "Reactant SMILES combination failed"
    reactant_smiles = Chem.MolToSmiles(combined)

    try:
        reactant_graph = gb.create_reactant_graph_with_metal(
            sub_smiles=str(substrate),
            bpin_smiles=str(bpin),
            ligand_smiles=str(ligand),
            catalyst_feature=catalyst_feature,
            sub_mark_indices_0_based=substrate_indices,
            global_smiles_for_fp=reactant_smiles,
            device=device,
            xtb_store=xtb_store,
        )
        product_graph = gb.create_product_graph(
            str(product), product_indices, device, xtb_store=xtb_store
        )
    except Exception as error:
        return None, f"Reaction graph error: {error}"
    if reactant_graph is None or product_graph is None:
        return None, "Reactant/product graph construction failed"

    reactant_global = build_global_features(
        ligand_fingerprint,
        reactant_smiles,
        nBits=fingerprint_nbits,
        ligand_smiles=str(ligand),
        xtb_global=reaction_xtb_vector(xtb_store, str(ligand), str(substrate), str(bpin), "sub"),
        **descriptor_settings,
    )
    product_global = build_global_features(
        ligand_fingerprint,
        str(product),
        nBits=fingerprint_nbits,
        ligand_smiles=str(ligand),
        xtb_global=reaction_xtb_vector(xtb_store, str(ligand), str(product), str(bpin), "prod"),
        **descriptor_settings,
    )
    if not gb.attach_global_to_graph(reactant_graph, reactant_global, device):
        return None, "Reactant global-feature construction failed"
    if not gb.attach_global_to_graph(product_graph, product_global, device):
        return None, "Product global-feature construction failed"
    return (name, environment_graph, reactant_graph, product_graph), None


@torch.no_grad()
def predict_dataframe(
    model,
    dataframe,
    device,
    batch_size=16,
    allow_missing_indices=False,
    indices_are_zero_based=False,
    default_catalyst=None,
    target_column="G (kcal/mol)",
    xtb_root=None,
):
    _check_required_columns(dataframe, default_catalyst=default_catalyst)
    fingerprint_nbits = _fingerprint_nbits_from_model(model)
    descriptor_settings = _descriptor_settings_from_model(model)
    xtb_store = get_xtb_store(xtb_root) if model.xtb_feature_config.get("enabled", True) else None
    built = []
    status = ["SKIP"] * len(dataframe)
    reasons = [""] * len(dataframe)
    for row_index in range(len(dataframe)):
        try:
            temperature_kelvin = temperature_kelvin_from_celsius(
                dataframe.iloc[row_index][TEMPERATURE_CELSIUS_COLUMN],
                row_index=row_index,
            )
        except ValueError as error:
            reasons[row_index] = str(error)
            continue
        graphs, error = _build_graphs_for_row(
            dataframe.iloc[row_index],
            device,
            fingerprint_nbits,
            descriptor_settings,
            allow_missing_indices=allow_missing_indices,
            indices_are_zero_based=indices_are_zero_based,
            default_catalyst=default_catalyst,
            xtb_store=xtb_store,
        )
        if graphs is None:
            reasons[row_index] = error or "Unknown graph error"
            continue
        name, environment, reactant, product = graphs
        built.append(
            (
                row_index,
                name,
                environment,
                reactant,
                product,
                temperature_kelvin,
            )
        )
        status[row_index] = "OK"
    if not built:
        raise RuntimeError("No valid samples were available for regression prediction.")

    predictions = [np.nan] * len(dataframe)
    temperatures_kelvin = [np.nan] * len(dataframe)
    enthalpies = [np.nan] * len(dataframe)
    entropies = [np.nan] * len(dataframe)
    model.eval()
    for start in range(0, len(built), int(batch_size)):
        batch = built[start : start + int(batch_size)]
        row_indices, environments, reactants, products, batch_temperatures = zip(
            *[(item[0], item[2], item[3], item[4], item[5]) for item in batch]
        )
        delta_g, delta_h, delta_s = model(
            dgl.batch(environments).to(device),
            dgl.batch(reactants).to(device),
            dgl.batch(products).to(device),
            torch.tensor(batch_temperatures, dtype=torch.float32, device=device),
            return_components=True,
        )
        for position, row_index in enumerate(row_indices):
            predictions[row_index] = float(delta_g[position].item())
            enthalpies[row_index] = float(delta_h[position].item())
            entropies[row_index] = float(delta_s[position].item())
            temperatures_kelvin[row_index] = float(batch_temperatures[position])
    result = dataframe.copy()
    result["Temperature (K)"] = temperatures_kelvin
    result["Predicted Delta H (kcal/mol)"] = enthalpies
    result["Predicted Delta S (cal/mol/K)"] = entropies
    result[f"Predicted {target_column}"] = predictions
    result["status"] = status
    result["reason"] = reasons
    for role, column in (("sub", "sub-type"), ("prod", "P"), ("lig", "L-type"), ("boron", "B-type")):
        result[f"xtb_{role}_available"] = [
            bool(xtb_store.get(str(dataframe.iloc[i][column]), role).available)
            if xtb_store is not None and status[i] == "OK" else False
            for i in range(len(dataframe))]
    result = attach_classification(result, default_catalyst=default_catalyst)
    return result, built


def load_regression_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint.get("metadata", {}) or {}
    else:
        state_dict = checkpoint
        metadata = {}
    model, config = build_regression_model_from_checkpoint(state_dict, metadata)
    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, metadata, config


def main():
    parser = argparse.ArgumentParser(description="Predict H/S -> G, then fixed Ir/Co interval classes.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="prediction_results.xlsx")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--default_catalyst", default="Ir")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--xtb_root", default=None)
    parser.add_argument("--allow_missing_indices", action="store_true")
    parser.add_argument("--indices_are_zero_based", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model, metadata, _ = load_regression_checkpoint(args.model, device)
    target_column = str(metadata.get("target_column", "G (kcal/mol)"))
    dataframe = pd.read_csv(args.input)
    result, _ = predict_dataframe(
        model,
        dataframe,
        device,
        batch_size=args.batch_size,
        allow_missing_indices=args.allow_missing_indices,
        indices_are_zero_based=args.indices_are_zero_based,
        default_catalyst=args.default_catalyst.strip() or None,
        target_column=target_column,
        xtb_root=args.xtb_root,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = export_classification(
        result, os.path.dirname(args.output) or ".",
        default_catalyst=args.default_catalyst.strip() or None,
        stem=os.path.splitext(os.path.basename(args.output))[0] + "_classification",
        role="User regression predictions",
    )
    if args.output.lower().endswith((".xlsx", ".xls")):
        result.to_excel(args.output, index=False)
    else:
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"Predicted {(result['status'] == 'OK').sum()}/{len(result)} rows -> "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
