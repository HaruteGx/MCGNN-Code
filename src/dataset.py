import ast
import math
import pandas as pd
from rdkit import Chem

from .graph_builders import (
    combine_smiles,
    create_environment_graph,
    create_reactant_graph_with_metal,
    create_product_graph,
    attach_global_to_graph,
)
from .descriptors import global_feature_dimension
from .mcencoding import encode_catalyst_feature, encode_ligand_fp, build_global_features
from .xtb_features import get_xtb_store, reaction_xtb_vector


SUBSTRATE_SITE_INDEX_COLUMN = "bpin_index_sub"
PRODUCT_SITE_INDEX_COLUMN = "bpin_index Production"
TEMPERATURE_CELSIUS_COLUMN = "T / C"
KELVIN_OFFSET = 273.15
REACTION_SITE_INDEX_FEATURE_LAYOUT = {
    "reactant_graph": {
        "source_column": SUBSTRATE_SITE_INDEX_COLUMN,
        "input_index_base": 1,
        "encoding": "set is_marked_reaction_site=1 on the indexed substrate atoms",
    },
    "product_graph": {
        "source_column": PRODUCT_SITE_INDEX_COLUMN,
        "input_index_base": 1,
        "encoding": "set is_marked_reaction_site=1 on the indexed product atoms",
    },
}


def temperature_kelvin_from_celsius(value, row_index=None):

    location = "" if row_index is None else f" at row {row_index}"
    try:
        temperature_celsius = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{TEMPERATURE_CELSIUS_COLUMN} must be numeric{location}; got {value!r}."
        )
    temperature_kelvin = temperature_celsius + KELVIN_OFFSET
    if not math.isfinite(temperature_kelvin) or temperature_kelvin <= 0:
        raise ValueError(
            f"{TEMPERATURE_CELSIUS_COLUMN} gives an invalid absolute temperature"
            f"{location}: {value!r}."
        )
    return temperature_kelvin


def parse_reaction_site_indices(value, column_name, row_index):

    if isinstance(value, (list, tuple)):
        parsed = list(value)
    else:
        if pd.isna(value) or str(value).strip() == "":
            raise ValueError(f"{column_name} is empty at row {row_index}.")
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (int, float)):
            parsed = [parsed]
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(
            f"{column_name} must be a non-empty index list at row {row_index}."
        )

    indices_0_based = []
    for raw_index in parsed:
        numeric_index = float(raw_index)
        integer_index = int(numeric_index)
        if not numeric_index.is_integer() or integer_index < 1:
            raise ValueError(
                f"{column_name} contains invalid 1-based index {raw_index!r} "
                f"at row {row_index}."
            )
        indices_0_based.append(integer_index - 1)
    if len(indices_0_based) != len(set(indices_0_based)):
        raise ValueError(
            f"{column_name} contains duplicated atom indices at row {row_index}."
        )
    return indices_0_based

def load_csv(data_path: str):
    return pd.read_csv(data_path)

def _resolve_catalysts(data: pd.DataFrame, default_catalyst=None):
    if 'Cat' in data.columns:
        catalysts = data['Cat'].tolist()
        if default_catalyst is not None:
            catalysts = [
                default_catalyst if pd.isna(value) or str(value).strip() == "" else value
                for value in catalysts
            ]
        return catalysts
    if default_catalyst is None:
        raise KeyError("Dataset has no 'Cat' column and no default_catalyst was provided.")
    return [default_catalyst] * len(data)


def filter_valid_indices(data: pd.DataFrame, default_catalyst=None):
    missing_index_columns = [
        column
        for column in (SUBSTRATE_SITE_INDEX_COLUMN, PRODUCT_SITE_INDEX_COLUMN)
        if column not in data.columns
    ]
    if missing_index_columns:
        raise KeyError(
            f"Dataset is missing reaction-site index columns: {missing_index_columns}"
        )
    names = data['Name'].tolist()
    solvents = data['sol'].tolist()
    substrate_smiles = data['sub-type'].tolist()
    ligand_info = data['L-type'].tolist()
    bpin_smiles = data['B-type'].tolist()
    product_smiles = data['P'].tolist()
    catalysts = _resolve_catalysts(data, default_catalyst=default_catalyst)

    valid = []
    for idx in range(len(data)):
        solvent = solvents[idx]
        sub = substrate_smiles[idx]
        bpin = bpin_smiles[idx]
        lig = ligand_info[idx]
        prod = product_smiles[idx]

        if catalysts[idx] not in ("Ir", "Co", "Ni"):
            print(f"Skip index {idx} due to unknown catalyst {catalysts[idx]}")
            continue

        molecule_values = {
            "sol": solvent,
            "sub-type": sub,
            "B-type": bpin,
            "L-type": lig,
            "P": prod,
        }
        molecules = {}
        invalid_columns = []
        for column, value in molecule_values.items():
            molecule = None if pd.isna(value) else Chem.MolFromSmiles(str(value))
            molecules[column] = molecule
            if molecule is None:
                invalid_columns.append(column)
        if invalid_columns:
            print(
                f"Skip index {idx} due to invalid SMILES columns: "
                f"{', '.join(invalid_columns)}"
            )
            continue


        reactant_mol = combine_smiles([sub, bpin, lig])
        if reactant_mol is None:
            print(f"Failed to combine SMILES at index {idx}: {sub}, {bpin}, {lig}")
            continue
        reactant_smiles = Chem.MolToSmiles(reactant_mol)
        if Chem.MolFromSmiles(reactant_smiles) is None:
            print(f"Invalid reactant SMILES at index {idx}: {reactant_smiles}")
            continue
        try:
            sub_indices = parse_reaction_site_indices(
                data.iloc[idx][SUBSTRATE_SITE_INDEX_COLUMN],
                SUBSTRATE_SITE_INDEX_COLUMN,
                idx,
            )
            product_indices = parse_reaction_site_indices(
                data.iloc[idx][PRODUCT_SITE_INDEX_COLUMN],
                PRODUCT_SITE_INDEX_COLUMN,
                idx,
            )
        except (ValueError, SyntaxError, TypeError) as error:
            print(f"Skip index {idx} due to invalid reaction-site indices: {error}")
            continue
        substrate_mol = molecules["sub-type"]
        product_mol = molecules["P"]
        if any(index >= substrate_mol.GetNumAtoms() for index in sub_indices):
            print(f"Skip index {idx}: {SUBSTRATE_SITE_INDEX_COLUMN} is out of bounds")
            continue
        if any(index >= product_mol.GetNumAtoms() for index in product_indices):
            print(f"Skip index {idx}: {PRODUCT_SITE_INDEX_COLUMN} is out of bounds")
            continue

        valid.append(idx)
    return valid

def build_dataset(
    data: pd.DataFrame,
    valid_indices,
    device,
    fp_nbits=2048,
    target_column='G (kcal/mol)',
    target_transform=None,
    default_catalyst=None,
    include_3d_descriptors=True,
    descriptor_num_conformers=3,
    descriptor_conformer_seed=20260822,
    optimize_conformers=True,
    progress_interval=50,
    use_xtb_features=True,
    xtb_root=None,
    xtb_store=None,
):
    xtb_store = (xtb_store or get_xtb_store(xtb_root)) if use_xtb_features else None
    global_feature_dim = global_feature_dimension(
        fp_nbits, include_3d=include_3d_descriptors
    )
    descriptor_kwargs = {
        "ligand_smiles": None,
        "include_3d_descriptors": bool(include_3d_descriptors),
        "descriptor_num_conformers": int(descriptor_num_conformers),
        "descriptor_conformer_seed": int(descriptor_conformer_seed),
        "optimize_conformers": bool(optimize_conformers),
    }

    names = data['Name'].tolist()
    solvents = data['sol'].tolist()
    substrate_smiles = data['sub-type'].tolist()
    ligand_info = data['L-type'].tolist()
    bpin_smiles = data['B-type'].tolist()
    product_smiles = data['P'].tolist()
    if target_column not in data.columns:
        raise KeyError(f"Dataset is missing target column: {target_column}")
    targets = data[target_column].tolist()
    bpin_index_sub = data[SUBSTRATE_SITE_INDEX_COLUMN].tolist()
    bpin_index_production = data[PRODUCT_SITE_INDEX_COLUMN].tolist()
    catalysts = _resolve_catalysts(data, default_catalyst=default_catalyst)
    if TEMPERATURE_CELSIUS_COLUMN not in data.columns:
        raise KeyError(
            f"Dataset is missing temperature column: {TEMPERATURE_CELSIUS_COLUMN}"
        )
    temperatures_celsius = data[TEMPERATURE_CELSIUS_COLUMN].tolist()

    dataset = []
    total_candidates = len(valid_indices)
    print(
        f"Building {total_candidates} graph samples on {device}; "
        f"3D descriptors={'on' if include_3d_descriptors else 'off'}.",
        flush=True,
    )
    for position, idx in enumerate(valid_indices, start=1):
        if (
            position == 1
            or position == total_candidates
            or (progress_interval and position % int(progress_interval) == 0)
        ):
            print(
                f"Graph build progress: {position}/{total_candidates} "
                f"(accepted {len(dataset)})",
                flush=True,
            )
        name = names[idx]

        solvent_smiles = solvents[idx]
        sub_smiles = substrate_smiles[idx]
        lig_smiles = ligand_info[idx]
        bpin_smiles_sample = bpin_smiles[idx]
        prod_smiles = product_smiles[idx]
        y = targets[idx]
        if pd.isna(y):
            print(f"Missing target '{target_column}' at idx {idx}")
            continue
        if target_transform is not None:
            y = target_transform(y)
        try:
            temperature_kelvin = temperature_kelvin_from_celsius(
                temperatures_celsius[idx], row_index=idx
            )
        except ValueError as error:
            print(f"Skip index {idx} due to invalid temperature: {error}")
            continue
        cat_id = catalysts[idx]
        try:
            sub_idx_0 = parse_reaction_site_indices(
                bpin_index_sub[idx], SUBSTRATE_SITE_INDEX_COLUMN, idx
            )
            prod_idx_0 = parse_reaction_site_indices(
                bpin_index_production[idx], PRODUCT_SITE_INDEX_COLUMN, idx
            )
        except (ValueError, SyntaxError, TypeError, IndexError) as e:
            print(f"Invalid atom index at idx {idx}: {e}")
            continue


        env_graph = create_environment_graph(
            solvent_smiles,
            device,
            include_3d_descriptors=include_3d_descriptors,
            descriptor_num_conformers=descriptor_num_conformers,
            descriptor_conformer_seed=descriptor_conformer_seed,
            optimize_conformers=optimize_conformers,
            xtb_store=xtb_store,
        )
        if env_graph is None:
            continue


        catalyst_feature = encode_catalyst_feature(cat_id)
        lig_fp = encode_ligand_fp(lig_smiles, nBits=fp_nbits)
        if catalyst_feature is None or lig_fp is None:
            continue

        reactant_mol = combine_smiles([sub_smiles, bpin_smiles_sample, lig_smiles])
        if reactant_mol is None:
            continue
        reactant_smiles_for_fp = Chem.MolToSmiles(reactant_mol)


        reactant_graph = create_reactant_graph_with_metal(
            sub_smiles=sub_smiles,
            bpin_smiles=bpin_smiles_sample,
            ligand_smiles=lig_smiles,
            catalyst_feature=catalyst_feature,
            sub_mark_indices_0_based=sub_idx_0,
            global_smiles_for_fp=reactant_smiles_for_fp,
            device=device,
            xtb_store=xtb_store,
        )
        if reactant_graph is None:
            continue

        descriptor_kwargs["ligand_smiles"] = lig_smiles
        global_react = build_global_features(
            lig_fp,
            reactant_smiles_for_fp,
            nBits=fp_nbits,
            xtb_global=reaction_xtb_vector(xtb_store, lig_smiles, sub_smiles,
                                          bpin_smiles_sample, "sub"),
            **descriptor_kwargs,
        )
        if global_react is None or not attach_global_to_graph(reactant_graph, global_react, device):
            continue


        product_graph = create_product_graph(prod_smiles, prod_idx_0, device, xtb_store=xtb_store)
        if product_graph is None:
            continue
        global_prod = build_global_features(
            lig_fp, prod_smiles, nBits=fp_nbits,
            xtb_global=reaction_xtb_vector(xtb_store, lig_smiles, prod_smiles,
                                          bpin_smiles_sample, "prod"),
            **descriptor_kwargs
        )
        if global_prod is None or not attach_global_to_graph(product_graph, global_prod, device):
            continue


        dataset.append(
            (name, env_graph, reactant_graph, product_graph, temperature_kelvin, y)
        )

    print(
        f"Graph construction complete: accepted {len(dataset)}/{total_candidates}.",
        flush=True,
    )
    if xtb_store is not None:
        print(f"xTB unique-molecule coverage: {xtb_store.report()['coverage']}", flush=True)
        if xtb_store.errors:
            print(f"WARNING: {len(xtb_store.errors)} xTB jobs could not be parsed; masks were used.", flush=True)
    return dataset, global_feature_dim


def get_class_values(data: pd.DataFrame, target_column='Class'):

    if target_column not in data.columns:
        raise KeyError(f"Dataset is missing class column: {target_column}")
    values = data[target_column].dropna().unique().tolist()
    values = [value.item() if hasattr(value, "item") else value for value in values]
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def build_classification_dataset(
    data: pd.DataFrame,
    valid_indices,
    device,
    fp_nbits=2048,
    target_column='Class',
    class_values=None,
    default_catalyst=None,
    include_3d_descriptors=True,
    descriptor_num_conformers=3,
    descriptor_conformer_seed=20260822,
    optimize_conformers=True,
):

    class_values = list(class_values or get_class_values(data, target_column=target_column))
    if len(class_values) < 2:
        raise ValueError(f"Classification requires at least two classes, got: {class_values}")
    class_to_index = {value: index for index, value in enumerate(class_values)}

    def encode_label(value):
        normalized = value.item() if hasattr(value, "item") else value
        if normalized not in class_to_index:
            raise ValueError(
                f"Unknown class label {normalized!r}; expected one of {class_values!r}"
            )
        return int(class_to_index[normalized])

    dataset, global_feature_dim = build_dataset(
        data=data,
        valid_indices=valid_indices,
        device=device,
        fp_nbits=fp_nbits,
        target_column=target_column,
        target_transform=encode_label,
        default_catalyst=default_catalyst,
        include_3d_descriptors=include_3d_descriptors,
        descriptor_num_conformers=descriptor_num_conformers,
        descriptor_conformer_seed=descriptor_conformer_seed,
        optimize_conformers=optimize_conformers,
    )
    return dataset, global_feature_dim, class_values
