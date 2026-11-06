import numpy as np
from rdkit import Chem

from .descriptors import molecular_descriptor_vector
from .features import NODE_FEATURE_DIM, atomic_reference_values, morgan_fp_bits
from .xtb_features import XTB_NODE_FEATURE_NAMES, XTB_REACTION_GLOBAL_NAMES


def encode_catalyst_feature(catalyst_identifier: str):

    catalyst_dict = {
        "Ir": {"atomic_number": 77},
        "Co": {"atomic_number": 27},
        "Ni": {"atomic_number": 28},
    }
    if catalyst_identifier not in catalyst_dict:
        return None
    catalyst = catalyst_dict[catalyst_identifier]
    atomic_number = catalyst["atomic_number"]
    reference = atomic_reference_values(atomic_number)
    feature = np.array(
        [
            atomic_number,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            reference["vdw_radius"],
            reference["covalent_radius"],
            reference["electronegativity"],
            reference["ionization_energy"],
            reference["d_electrons"],
            0.0,
            reference["atomic_mass_scaled"],
        ] + [0.0] * len(XTB_NODE_FEATURE_NAMES) + [1.0, 0.0],
        dtype=np.float32,
    )
    if feature.shape != (NODE_FEATURE_DIM,):
        raise RuntimeError("Catalyst node feature schema is inconsistent.")
    return feature


def encode_ligand_fp(ligand_smiles: str, nBits=2048):
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        return None
    return morgan_fp_bits(mol, radius=2, nBits=nBits)


def build_global_features(
    ligand_fp: np.ndarray,
    molecule_smiles: str,
    nBits=2048,
    ligand_smiles=None,
    include_3d_descriptors=True,
    descriptor_num_conformers=3,
    descriptor_conformer_seed=20260822,
    optimize_conformers=True,
    xtb_global=None,
):

    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None or ligand_smiles is None:
        return None
    molecule_fp = morgan_fp_bits(mol, radius=2, nBits=nBits)
    if len(ligand_fp) != int(nBits):
        raise ValueError(
            f"Ligand fingerprint has {len(ligand_fp)} bits; expected {int(nBits)}."
        )
    descriptor_kwargs = {
        "include_3d": bool(include_3d_descriptors),
        "max_conformers": int(descriptor_num_conformers),
        "conformer_seed": int(descriptor_conformer_seed),
        "optimize_conformers": bool(optimize_conformers),
    }
    ligand_descriptors = molecular_descriptor_vector(
        ligand_smiles, **descriptor_kwargs
    )
    molecule_descriptors = molecular_descriptor_vector(
        molecule_smiles, **descriptor_kwargs
    )
    if ligand_descriptors is None or molecule_descriptors is None:
        return None
    if xtb_global is None:
        xtb_global = np.zeros(len(XTB_REACTION_GLOBAL_NAMES), dtype=np.float32)
    if np.asarray(xtb_global).shape != (len(XTB_REACTION_GLOBAL_NAMES),):
        raise ValueError("Wrong xTB reaction-global feature dimension")
    return np.concatenate(
        [ligand_fp, molecule_fp, ligand_descriptors, molecule_descriptors, xtb_global]
    ).astype(np.float32)
