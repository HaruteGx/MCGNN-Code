import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .xtb_features import XTB_NODE_FEATURE_NAMES, XTB_EDGE_FEATURE_NAMES


PAULING_ELECTRONEGATIVITY = {
    1: 2.20,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    27: 1.88,
    28: 1.91,
    35: 2.96,
    53: 2.66,
    77: 2.20,
}
FIRST_IONIZATION_ENERGY_EV = {
    1: 13.5984,
    5: 8.2980,
    6: 11.2603,
    7: 14.5341,
    8: 13.6181,
    9: 17.4228,
    14: 8.1517,
    15: 10.4867,
    16: 10.3600,
    17: 12.9676,
    27: 7.8810,
    28: 7.6400,
    35: 11.8138,
    53: 10.4513,
    77: 8.9670,
}
NEUTRAL_ATOM_D_ELECTRON_COUNT = {27: 7.0, 28: 8.0, 77: 7.0}


NODE_FEATURE_NAMES = (
    "atomic_number",
    "coordination_number",
    "hybridization_code",
    "is_aromatic",
    "is_marked_reaction_site",
    "formal_charge",
    "total_valence",
    "total_hydrogen_count",
    "is_in_ring",
    "chiral_tag_code",
    "vdw_radius_angstrom",
    "covalent_radius_angstrom",
    "pauling_electronegativity",
    "first_ionization_energy_ev",
    "neutral_atom_d_electron_count",
    "gasteiger_partial_charge",
    "atomic_mass_div_200",
) + XTB_NODE_FEATURE_NAMES + ("is_metal", "is_potential_ligand_donor")
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
REACTION_SITE_NODE_FEATURE_INDEX = NODE_FEATURE_NAMES.index(
    "is_marked_reaction_site"
)

EDGE_FEATURE_NAMES = (
    "bond_type_single",
    "bond_type_double",
    "bond_type_triple",
    "bond_type_aromatic",
    "bond_order",
    "is_conjugated",
    "is_in_ring",
    "stereo_code",
    "endpoint_electronegativity_difference",
    "endpoint_covalent_radius_sum_angstrom",
) + XTB_EDGE_FEATURE_NAMES
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)
METAL_NODE_FEATURE_INDEX = NODE_FEATURE_NAMES.index("is_metal")
DONOR_NODE_FEATURE_INDEX = NODE_FEATURE_NAMES.index("is_potential_ligand_donor")


def mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(smiles)


def atomic_reference_values(atomic_number):
    periodic_table = Chem.GetPeriodicTable()
    atomic_number = int(atomic_number)
    return {
        "vdw_radius": float(periodic_table.GetRvdw(atomic_number)),
        "covalent_radius": float(periodic_table.GetRcovalent(atomic_number)),
        "electronegativity": float(
            PAULING_ELECTRONEGATIVITY.get(atomic_number, 0.0)
        ),
        "ionization_energy": float(
            FIRST_IONIZATION_ENERGY_EV.get(atomic_number, 0.0)
        ),
        "d_electrons": float(
            NEUTRAL_ATOM_D_ELECTRON_COUNT.get(atomic_number, 0.0)
        ),
        "atomic_mass_scaled": float(periodic_table.GetAtomicWeight(atomic_number)) / 200.0,
    }


def _gasteiger_charge_map(mol):
    charged = Chem.Mol(mol)
    try:
        AllChem.ComputeGasteigerCharges(charged)
    except Exception:
        return [0.0] * charged.GetNumAtoms()
    charges = []
    for atom in charged.GetAtoms():
        raw = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        charges.append(value if np.isfinite(value) else 0.0)
    return charges


def generate_node_features_from_mol(mol: Chem.Mol, atom_indices_0_based=None,
                                    xtb_features=None, is_ligand=False):

    if mol is None:
        return None
    marked = set(atom_indices_0_based) if atom_indices_0_based is not None else set()
    charges = _gasteiger_charge_map(mol)
    features = []
    for index, atom in enumerate(mol.GetAtoms()):
        atomic_number = atom.GetAtomicNum()
        reference = atomic_reference_values(atomic_number)
        features.append(
            [
                atomic_number,
                atom.GetDegree(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                1 if index in marked else 0,
                atom.GetFormalCharge(),
                atom.GetTotalValence(),
                atom.GetTotalNumHs(includeNeighbors=True),
                int(atom.IsInRing()),
                int(atom.GetChiralTag()),
                reference["vdw_radius"],
                reference["covalent_radius"],
                reference["electronegativity"],
                reference["ionization_energy"],
                reference["d_electrons"],
                charges[index],
                reference["atomic_mass_scaled"],
            ] + (xtb_features.atom[index].tolist() if xtb_features is not None
                 else [0.0] * len(XTB_NODE_FEATURE_NAMES))
            + [0.0, float(is_ligand and atomic_number in (7, 15))]
        )
    return features


def _bond_type_one_hot(bond_type):
    one_hot = [0.0, 0.0, 0.0, 0.0]
    if bond_type == Chem.rdchem.BondType.SINGLE:
        one_hot[0] = 1.0
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        one_hot[1] = 1.0
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        one_hot[2] = 1.0
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        one_hot[3] = 1.0
    return one_hot


def generate_edge_features_from_mol(mol: Chem.Mol, xtb_features=None):

    if mol is None:
        return None, None, None
    edge_features, source, destination = [], [], []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_reference = atomic_reference_values(begin.GetAtomicNum())
        end_reference = atomic_reference_values(end.GetAtomicNum())
        feature = _bond_type_one_hot(bond.GetBondType()) + [
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(int(bond.GetStereo())),
            abs(
                begin_reference["electronegativity"]
                - end_reference["electronegativity"]
            ),
            begin_reference["covalent_radius"]
            + end_reference["covalent_radius"],
        ]
        begin_index, end_index = begin.GetIdx(), end.GetIdx()
        feature += (xtb_features.edge_vector(begin_index, end_index)
                    if xtb_features is not None else [0.0] * len(XTB_EDGE_FEATURE_NAMES))
        source.extend((begin_index, end_index))
        destination.extend((end_index, begin_index))
        edge_features.extend((feature, feature))
    return edge_features, source, destination


def morgan_fp_bits(mol: Chem.Mol, radius=2, nBits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array([int(x) for x in fp.ToBitString()], dtype=np.float32)


def find_first_atom_index_by_symbol(mol: Chem.Mol, symbol: str):
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol:
            return atom.GetIdx()
    return None


def metal_edge_feature(other_atomic_number, metal_atomic_number):

    other = atomic_reference_values(other_atomic_number)
    metal = atomic_reference_values(metal_atomic_number)
    return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            abs(other["electronegativity"] - metal["electronegativity"]),
            other["covalent_radius"] + metal["covalent_radius"]] + [0.0] * 5 + [1.0]
