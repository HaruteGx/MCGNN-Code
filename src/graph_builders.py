import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem

from src.descriptors import molecular_descriptor_vector
from .xtb_features import XTB_GLOBAL_DIM
from .features import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    mol_from_smiles,
    generate_node_features_from_mol,
    generate_edge_features_from_mol,
    find_first_atom_index_by_symbol,
    metal_edge_feature,
)

def create_environment_graph(
    solvent_smiles,
    device,
    include_3d_descriptors=True,
    descriptor_num_conformers=3,
    descriptor_conformer_seed=20260822,
    optimize_conformers=True,
    xtb_store=None,
):


    mol = mol_from_smiles(solvent_smiles)
    if mol is None:
        print(f"Invalid solvent SMILES: {solvent_smiles}")
        return None

    xtb = xtb_store.get(solvent_smiles, "sol") if xtb_store is not None else None
    node_feats = generate_node_features_from_mol(mol, xtb_features=xtb)
    edge_feats, src, dst = generate_edge_features_from_mol(mol, xtb_features=xtb)
    if node_feats is None or edge_feats is None:
        return None

    g = dgl.graph((src, dst), num_nodes=len(node_feats)).to(device)
    g.ndata["node"] = torch.tensor(node_feats, dtype=torch.float32, device=device)
    g.edata["edge"] = torch.tensor(
        edge_feats, dtype=torch.float32, device=device
    ).reshape(-1, EDGE_FEATURE_DIM)

    fp256 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    solvent_feature = np.array([int(bit) for bit in fp256.ToBitString()], dtype=np.float32)

    solvent_descriptors = molecular_descriptor_vector(
        solvent_smiles,
        include_3d=include_3d_descriptors,
        max_conformers=descriptor_num_conformers,
        conformer_seed=descriptor_conformer_seed,
        optimize_conformers=optimize_conformers,
    )
    if solvent_descriptors is None:
        return None
    global_vec = np.concatenate(
        [solvent_feature, solvent_descriptors,
         xtb.global_vector if xtb is not None else np.zeros(XTB_GLOBAL_DIM)]
    ).astype(np.float32, copy=False)
    global_tensor = torch.tensor(global_vec, dtype=torch.float32, device=device).unsqueeze(0).repeat(g.num_nodes(), 1)
    g.ndata["global"] = global_tensor
    return g

def _append_component(mol, node_feats_all, edge_feats_all, src_all, dst_all, offset,
                      mark_indices_0_based=None, xtb_features=None, is_ligand=False):
    node_feats = generate_node_features_from_mol(
        mol, atom_indices_0_based=mark_indices_0_based,
        xtb_features=xtb_features, is_ligand=is_ligand)
    edge_feats, src, dst = generate_edge_features_from_mol(mol, xtb_features=xtb_features)

    n = len(node_feats)
    node_feats_all.extend(node_feats)


    for s, t, ef in zip(src, dst, edge_feats):
        src_all.append(s + offset)
        dst_all.append(t + offset)
        edge_feats_all.append(ef)

    return n

def create_reactant_graph_with_metal(sub_smiles, bpin_smiles, ligand_smiles,
                                    catalyst_feature, sub_mark_indices_0_based,
                                    global_smiles_for_fp, device, xtb_store=None):


    sub_mol = mol_from_smiles(sub_smiles)
    bpin_mol = mol_from_smiles(bpin_smiles)
    lig_mol = mol_from_smiles(ligand_smiles)
    if sub_mol is None or bpin_mol is None or lig_mol is None:
        print("Invalid SMILES in reactant components.")
        return None
    if len(catalyst_feature) != NODE_FEATURE_DIM:
        raise ValueError(
            f"Catalyst feature has {len(catalyst_feature)} values; "
            f"expected {NODE_FEATURE_DIM}."
        )


    n_sub_atoms = sub_mol.GetNumAtoms()
    if any((i < 0 or i >= n_sub_atoms) for i in sub_mark_indices_0_based):
        print(f"Invalid marked indices for substrate: {sub_mark_indices_0_based}, n_sub_atoms={n_sub_atoms}")
        return None

    node_feats_all, edge_feats_all, src_all, dst_all = [], [], [], []
    offset = 0


    n_sub = _append_component(sub_mol, node_feats_all, edge_feats_all, src_all, dst_all, offset,
                             mark_indices_0_based=sub_mark_indices_0_based,
                             xtb_features=xtb_store.get(sub_smiles, "sub") if xtb_store is not None else None)
    offset += n_sub


    n_bpin = _append_component(bpin_mol, node_feats_all, edge_feats_all, src_all, dst_all, offset,
                              mark_indices_0_based=None,
                              xtb_features=xtb_store.get(bpin_smiles, "boron") if xtb_store is not None else None)
    bpin_offset = offset
    offset += n_bpin


    n_lig = _append_component(lig_mol, node_feats_all, edge_feats_all, src_all, dst_all, offset,
                              mark_indices_0_based=None, is_ligand=True,
                              xtb_features=xtb_store.get(ligand_smiles, "lig") if xtb_store is not None else None)
    offset += n_lig


    metal_node_id = offset
    node_feats_all.append([float(x) for x in catalyst_feature])
    offset += 1


    metal_atomic_number = int(catalyst_feature[0])
    for c_idx in sub_mark_indices_0_based:
        mef = metal_edge_feature(
            sub_mol.GetAtomWithIdx(c_idx).GetAtomicNum(), metal_atomic_number
        )
        src_all.append(c_idx);        dst_all.append(metal_node_id); edge_feats_all.append(mef)
        src_all.append(metal_node_id); dst_all.append(c_idx);        edge_feats_all.append(mef)


    b_idx_local = find_first_atom_index_by_symbol(bpin_mol, "B")
    if b_idx_local is not None:
        b_idx_global = bpin_offset + b_idx_local
        mef = metal_edge_feature(
            bpin_mol.GetAtomWithIdx(b_idx_local).GetAtomicNum(),
            metal_atomic_number,
        )
        src_all.append(b_idx_global);  dst_all.append(metal_node_id); edge_feats_all.append(mef)
        src_all.append(metal_node_id); dst_all.append(b_idx_global);  edge_feats_all.append(mef)


    g = dgl.graph((src_all, dst_all), num_nodes=len(node_feats_all)).to(device)
    g.ndata["node"] = torch.tensor(node_feats_all, dtype=torch.float32, device=device)
    g.edata["edge"] = torch.tensor(
        edge_feats_all, dtype=torch.float32, device=device
    ).reshape(-1, EDGE_FEATURE_DIM)


    return g

def create_product_graph(product_smiles, mark_indices_0_based, device, xtb_store=None):
    mol = mol_from_smiles(product_smiles)
    if mol is None:
        print(f"Invalid product SMILES: {product_smiles}")
        return None

    n_atoms = mol.GetNumAtoms()
    if any((i < 0 or i >= n_atoms) for i in mark_indices_0_based):
        print(f"Invalid marked indices for product: {mark_indices_0_based}, n_atoms={n_atoms}")
        return None

    xtb = xtb_store.get(product_smiles, "prod") if xtb_store is not None else None
    node_feats = generate_node_features_from_mol(
        mol, atom_indices_0_based=mark_indices_0_based, xtb_features=xtb)
    edge_feats, src, dst = generate_edge_features_from_mol(mol, xtb_features=xtb)

    g = dgl.graph((src, dst), num_nodes=len(node_feats)).to(device)
    g.ndata["node"] = torch.tensor(node_feats, dtype=torch.float32, device=device)
    g.edata["edge"] = torch.tensor(
        edge_feats, dtype=torch.float32, device=device
    ).reshape(-1, EDGE_FEATURE_DIM)
    return g

def attach_global_to_graph(graph, global_vec_1d, device):


    if graph is None or global_vec_1d is None:
        return False
    gv = torch.tensor(global_vec_1d, dtype=torch.float32, device=device).unsqueeze(0)
    graph.ndata["global"] = gv.repeat(graph.num_nodes(), 1)
    return True

def combine_smiles(smiles_list):


    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    if not mols:
        return None
    combined = mols[0]
    for m in mols[1:]:
        combined = Chem.CombineMols(combined, m)
    return combined
