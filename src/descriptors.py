from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from .xtb_features import XTB_GLOBAL_FEATURE_NAMES, XTB_REACTION_GLOBAL_NAMES

try:
    from rdkit.Chem import rdFreeSASA
except ImportError:
    rdFreeSASA = None


FAST_2D_DESCRIPTOR_NAMES = (
    "molecular_weight_div_500",
    "rotatable_bond_count_div_20",
    "ring_count_div_10",
    "flexibility_rotatable_per_heavy_atom",
    "bertz_topological_complexity_div_2000",
    "topological_symmetry_fraction",
    "labute_asa_2d_div_500",
    "molar_refractivity_div_200",
    "fraction_csp3",
    "formal_charge_div_5",
    "heavy_atom_count_div_100",
    "phosphorus_atom_count_div_10",
    "nitrogen_atom_count_div_20",
    "pn_gasteiger_charge_sum",
    "pn_gasteiger_charge_mean",
    "pn_gasteiger_charge_min",
    "pn_gasteiger_charge_max",
)

CONFORMER_3D_DESCRIPTOR_NAMES = (
    "rdkit_embedded_conformer_fraction",
    "rdkit_sasa_mean_div_500",
    "rdkit_sasa_std_div_500",
    "rdkit_volume_mean_div_1000",
    "rdkit_volume_std_div_1000",
    "rdkit_radius_of_gyration_mean_div_10",
    "rdkit_radius_of_gyration_std_div_10",
    "rdkit_asphericity_mean",
    "rdkit_asphericity_std",
    "rdkit_plane_of_best_fit_mean",
    "rdkit_plane_of_best_fit_std",
    "rdkit_spherocity_mean",
    "rdkit_spherocity_std",
)

EXCLUDED_NON_FAST_FEATURES = {
    "vibrational_frequencies": (
        "requires a Hessian from a defined 3D structure and force field or "
        "electronic-structure method"
    ),
    "low_frequency_modes": "same requirement as vibrational frequencies",
    "oxidation_state": "cannot be assigned reliably from the current SMILES columns",
    "spin_state": "not encoded by the current SMILES/catalyst identifier",
    "ligand_donor_strength": "requires an external experimental or calculated scale",
    "buried_volume": "requires a metal-ligand 3D complex and donor-axis definition",
    "cone_angle": "requires a metal-ligand 3D complex and metal-ligand axis",
    "bite_angle": "requires a coordinated bidentate 3D complex",
    "sterimol_B1_B5_L": "requires a selected conformer and a defined measurement axis",
    "HOMO_LUMO": "orbital energies require an electronic-structure calculation",
}


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _gasteiger_charges(mol):
    charged = Chem.Mol(mol)
    try:
        AllChem.ComputeGasteigerCharges(charged)
    except Exception:
        return np.zeros(charged.GetNumAtoms(), dtype=np.float32)
    values = []
    for atom in charged.GetAtoms():
        value = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        values.append(_finite(value))
    return np.asarray(values, dtype=np.float32)


@lru_cache(maxsize=32768)
def _fast_2d_from_canonical(canonical_smiles):
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return tuple(0.0 for _ in FAST_2D_DESCRIPTOR_NAMES)

    heavy_atoms = int(mol.GetNumHeavyAtoms())
    rotatable = int(Lipinski.NumRotatableBonds(mol))
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    symmetry_fraction = (
        1.0 - len(set(ranks)) / len(ranks) if ranks else 0.0
    )
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    phosphorus_count = sum(atom.GetAtomicNum() == 15 for atom in mol.GetAtoms())
    nitrogen_count = sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms())

    charges = _gasteiger_charges(mol)
    pn_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in (7, 15)
    ]
    pn_charges = charges[pn_indices] if pn_indices else np.zeros(0, dtype=np.float32)
    if pn_charges.size:
        charge_stats = (
            float(pn_charges.sum()),
            float(pn_charges.mean()),
            float(pn_charges.min()),
            float(pn_charges.max()),
        )
    else:
        charge_stats = (0.0, 0.0, 0.0, 0.0)

    values = (
        Descriptors.MolWt(mol) / 500.0,
        rotatable / 20.0,
        rings / 10.0,
        rotatable / max(heavy_atoms, 1),
        Descriptors.BertzCT(mol) / 2000.0,
        symmetry_fraction,
        rdMolDescriptors.CalcLabuteASA(mol) / 500.0,
        Descriptors.MolMR(mol) / 200.0,
        rdMolDescriptors.CalcFractionCSP3(mol),
        formal_charge / 5.0,
        heavy_atoms / 100.0,
        phosphorus_count / 10.0,
        nitrogen_count / 20.0,
        *charge_stats,
    )
    return tuple(_finite(value) for value in values)


def fast_2d_descriptor_vector(smiles):
    canonical = _canonical_smiles(smiles)
    if canonical is None:
        return None
    return np.asarray(_fast_2d_from_canonical(canonical), dtype=np.float32)


def _optimize_conformers(mol, conformer_ids):
    if not conformer_ids:
        return
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMoleculeConfs(
                mol, numThreads=1, maxIters=100, mmffVariant="MMFF94s"
            )
        elif AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=100)
    except Exception:

        pass


@lru_cache(maxsize=32768)
def _single_fragment_3d_statistics(canonical_smiles, max_conformers, seed, optimize):
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = 0.50
    params.numThreads = 1
    try:
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(
                mol, numConfs=max(1, int(max_conformers)), params=params
            )
        )
    except Exception:
        conformer_ids = []
    if not conformer_ids:
        return None
    if optimize:
        _optimize_conformers(mol, conformer_ids)

    periodic_table = Chem.GetPeriodicTable()
    sasa_radii = [
        max(float(periodic_table.GetRvdw(atom.GetAtomicNum())), 0.1)
        for atom in mol.GetAtoms()
    ]
    values = []
    for conformer_id in conformer_ids:
        try:
            sasa = (
                rdFreeSASA.CalcSASA(mol, sasa_radii, confIdx=int(conformer_id))
                if rdFreeSASA is not None
                else 0.0
            )
            values.append(
                (
                    sasa,
                    AllChem.ComputeMolVolume(mol, confId=int(conformer_id)),
                    rdMolDescriptors.CalcRadiusOfGyration(mol, confId=int(conformer_id)),
                    rdMolDescriptors.CalcAsphericity(mol, confId=int(conformer_id)),
                    rdMolDescriptors.CalcPBF(mol, confId=int(conformer_id)),
                    rdMolDescriptors.CalcSpherocityIndex(mol, confId=int(conformer_id)),
                )
            )
        except Exception:
            continue
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.shape[0]),
        "heavy_atoms": max(int(Chem.RemoveHs(mol).GetNumHeavyAtoms()), 1),
        "mean": np.nan_to_num(array.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0),
        "std": np.nan_to_num(array.std(axis=0), nan=0.0, posinf=0.0, neginf=0.0),
    }


@lru_cache(maxsize=32768)
def _conformer_3d_from_canonical(canonical_smiles, max_conformers, seed, optimize):
    parent = Chem.MolFromSmiles(canonical_smiles)
    if parent is None:
        return tuple(0.0 for _ in CONFORMER_3D_DESCRIPTOR_NAMES)
    fragments = Chem.GetMolFrags(parent, asMols=True, sanitizeFrags=True)
    statistics = []
    for fragment_index, fragment in enumerate(fragments):
        fragment_smiles = Chem.MolToSmiles(
            fragment, canonical=True, isomericSmiles=True
        )
        item = _single_fragment_3d_statistics(
            fragment_smiles,
            int(max_conformers),
            int(seed) + 104729 * fragment_index,
            bool(optimize),
        )
        if item is not None:
            statistics.append(item)
    if not statistics:
        return tuple(0.0 for _ in CONFORMER_3D_DESCRIPTOR_NAMES)

    total_heavy = float(sum(item["heavy_atoms"] for item in statistics))
    weights = np.asarray(
        [item["heavy_atoms"] / total_heavy for item in statistics], dtype=float
    )
    means = np.asarray([item["mean"] for item in statistics], dtype=float)
    stds = np.asarray([item["std"] for item in statistics], dtype=float)


    sasa_mean, volume_mean = means[:, 0].sum(), means[:, 1].sum()
    sasa_std = float(np.sqrt(np.square(stds[:, 0]).sum()))
    volume_std = float(np.sqrt(np.square(stds[:, 1]).sum()))

    shape_mean = (means[:, 2:] * weights[:, None]).sum(axis=0)
    shape_std = (stds[:, 2:] * weights[:, None]).sum(axis=0)
    conformer_fraction = float(
        np.mean([item["count"] / max(int(max_conformers), 1) for item in statistics])
    )
    values = (
        conformer_fraction,
        sasa_mean / 500.0,
        sasa_std / 500.0,
        volume_mean / 1000.0,
        volume_std / 1000.0,
        shape_mean[0] / 10.0,
        shape_std[0] / 10.0,
        shape_mean[1],
        shape_std[1],
        shape_mean[2],
        shape_std[2],
        shape_mean[3],
        shape_std[3],
    )
    return tuple(_finite(value) for value in values)


def conformer_3d_descriptor_vector(
    smiles, max_conformers=3, seed=20260822, optimize=True
):
    canonical = _canonical_smiles(smiles)
    if canonical is None:
        return None
    return np.asarray(
        _conformer_3d_from_canonical(
            canonical, int(max_conformers), int(seed), bool(optimize)
        ),
        dtype=np.float32,
    )


def molecular_descriptor_names(include_3d=True):
    names = list(FAST_2D_DESCRIPTOR_NAMES)
    if include_3d:
        names.extend(CONFORMER_3D_DESCRIPTOR_NAMES)
    return tuple(names)


def molecular_descriptor_vector(
    smiles,
    include_3d=True,
    max_conformers=3,
    conformer_seed=20260822,
    optimize_conformers=True,
):
    two_d = fast_2d_descriptor_vector(smiles)
    if two_d is None:
        return None
    if not include_3d:
        return two_d
    three_d = conformer_3d_descriptor_vector(
        smiles,
        max_conformers=max_conformers,
        seed=conformer_seed,
        optimize=optimize_conformers,
    )
    if three_d is None:
        three_d = np.zeros(len(CONFORMER_3D_DESCRIPTOR_NAMES), dtype=np.float32)
    return np.concatenate([two_d, three_d]).astype(np.float32)


def global_descriptor_names(include_3d=True):
    per_molecule = molecular_descriptor_names(include_3d=include_3d)
    return tuple(
        [f"ligand_{name}" for name in per_molecule]
        + [f"molecule_{name}" for name in per_molecule]
        + list(XTB_REACTION_GLOBAL_NAMES)
    )


def global_feature_dimension(fingerprint_nbits, include_3d=True):
    return 2 * int(fingerprint_nbits) + len(global_descriptor_names(include_3d))


def environment_global_feature_dimension(fingerprint_nbits=256, include_3d=True):

    return (int(fingerprint_nbits) + len(molecular_descriptor_names(include_3d))
            + len(XTB_GLOBAL_FEATURE_NAMES))


def global_descriptor_config(
    include_3d=True,
    max_conformers=3,
    conformer_seed=20260822,
    optimize_conformers=True,
):
    return {
        "fast_2d_enabled": True,
        "include_3d": bool(include_3d),
        "max_conformers": int(max_conformers),
        "conformer_seed": int(conformer_seed),
        "optimize_conformers": bool(optimize_conformers),
        "per_molecule_descriptor_names": list(
            molecular_descriptor_names(include_3d=include_3d)
        ),
        "global_descriptor_names": list(
            global_descriptor_names(include_3d=include_3d)
        ),
        "environment_descriptor_names": [
            f"solvent_{name}"
            for name in molecular_descriptor_names(include_3d=include_3d)
        ] + [f"solvent_{name}" for name in XTB_GLOBAL_FEATURE_NAMES],
        "three_d_method": (
            "ETKDGv3 + MMFF94s/UFF relaxation; independent fragment aggregation"
            if include_3d
            else "disabled"
        ),
        "excluded_features": {key: value for key, value in EXCLUDED_NON_FAST_FEATURES.items()
                              if key not in {"vibrational_frequencies", "low_frequency_modes", "HOMO_LUMO"}},
    }
