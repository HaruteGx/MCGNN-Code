import numpy as np
import torch
from rdkit import Chem

from src.features import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


def substrate_mapping(graph, smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid substrate SMILES")
    n = mol.GetNumAtoms()
    if graph.num_nodes() < n:
        raise ValueError("Substrate has more atoms than the reactant graph")
    np.testing.assert_array_equal(graph.ndata["node"][:n, 0].cpu().numpy(),
                                  [a.GetAtomicNum() for a in mol.GetAtoms()])
    source, target = (x.cpu().tolist() for x in graph.edges(order="eid"))
    lookup = {}
    for eid, (u, v) in enumerate(zip(source, target)):
        if u < n and v < n:
            lookup.setdefault((u, v), []).append(eid)
    pairs = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if len(lookup.get((u, v), [])) != 1 or len(lookup.get((v, u), [])) != 1:
            raise ValueError("Every substrate bond must map to exactly two directed edges")
        pair = [lookup[(u, v)][0], lookup[(v, u)][0]]
        np.testing.assert_allclose(graph.edata["edge"][pair, 4].cpu(), bond.GetBondTypeAsDouble())
        pairs.append(pair)
    if len(lookup) != 2 * mol.GetNumBonds():
        raise ValueError("Unexpected substrate edges; cannot map chemical bonds safely")
    return mol, pairs


def reference_features(graph, mol, pairs):

    original = (graph.ndata["node"].detach().clone(), graph.edata["edge"].detach().clone())
    base = tuple(x.clone() for x in original)
    fixed_nodes = {"is_marked_reaction_site", "is_metal", "is_potential_ligand_donor"}
    node_columns = [i for i, name in enumerate(NODE_FEATURE_NAMES)
                    if name not in fixed_nodes and not name.endswith("_available")]
    edge_columns = [i for i, name in enumerate(EDGE_FEATURE_NAMES)
                    if not name.endswith("_available") and name != "is_virtual_metal_edge"]
    for index in range(mol.GetNumAtoms()):
        base[0][index, node_columns] = 0
    for pair in pairs:
        for eid in pair:
            base[1][eid, edge_columns] = 0
    return original, base


def integrated_gradients(score, inputs, baseline, steps=32, max_steps=512,
                         atol=.002, rtol=.02):


    if steps < 2 or max_steps < 2 * steps:
        raise ValueError("Need at least two quadrature resolutions")
    delta = tuple(x - b for x, b in zip(inputs, baseline))
    with torch.no_grad():
        endpoint, reference = float(score(*inputs)), float(score(*baseline))
    previous, history = None, []
    count = steps
    while True:
        points, weights = np.polynomial.legendre.leggauss(count)
        total = [torch.zeros_like(x) for x in inputs]
        for point, weight in zip((points + 1) / 2, weights / 2):
            current = tuple((b + float(point) * d).detach().requires_grad_(True)
                            for b, d in zip(baseline, delta))
            value = score(*current)
            grads = torch.autograd.grad(value, current, allow_unused=True)
            for i, grad in enumerate(grads):
                if grad is not None:
                    total[i].add_(grad.detach(), alpha=float(weight))
        attribution = tuple(d * grad for d, grad in zip(delta, total))
        flat = torch.cat([x.flatten() for x in attribution])
        residual = abs(float(flat.sum()) - (endpoint - reference))
        change = None if previous is None else float(torch.abs(flat - previous).sum())
        l1 = float(flat.abs().sum())
        converged = (change is not None and change <= atol + rtol * l1
                     and residual <= atol + rtol * abs(endpoint-reference))
        history.append(dict(steps=count, completeness_abs_error=residual,
                            attribution_l1_change=change, attribution_l1=l1))
        if converged or count >= max_steps:
            break
        previous = flat
        count = min(2 * count, max_steps)
    audit = dict(predicted_delta_g=endpoint, reference_delta_g=reference,
                 predicted_minus_reference=endpoint-reference,
                 attribution_sum=float(flat.sum()), converged=converged,
                 atol=atol, rtol=rtol, integration_history=history)
    return attribution, audit


def explain(model, sample, smiles, steps=32, max_steps=512):
    _, environment, reactant, product, temperature, _ = sample
    mol, pairs = substrate_mapping(reactant, smiles)
    inputs, base = reference_features(reactant, mol, pairs)

    def score(nodes, edges):
        with reactant.local_scope():
            reactant.ndata["node"] = nodes
            reactant.edata["edge"] = edges
            return model(environment, reactant, product, [temperature]).reshape(())

    attribution, audit = integrated_gradients(score, inputs, base, steps, max_steps)
    node, edge = (x.detach().cpu().numpy() for x in attribution)
    atom_scores = node[:mol.GetNumAtoms()].sum(axis=1)

    bond_scores = np.array([edge[pair].sum() for pair in pairs])
    np.testing.assert_allclose(atom_scores.sum() + bond_scores.sum(), audit["attribution_sum"], atol=2e-5)
    audit["n_atoms"], audit["n_bonds"] = mol.GetNumAtoms(), mol.GetNumBonds()
    audit["predicted_class"] = int(np.searchsorted([23, 25], audit["predicted_delta_g"], side="right"))
    audit["reference_class"] = int(np.searchsorted([23, 25], audit["reference_delta_g"], side="right"))
    return mol, pairs, atom_scores, bond_scores, node, edge, audit
