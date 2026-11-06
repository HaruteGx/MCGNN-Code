from src.descriptors import molecular_descriptor_names, FAST_2D_DESCRIPTOR_NAMES
from src.xtb_features import XTB_GLOBAL_FEATURE_NAMES, XTB_REACTION_GLOBAL_NAMES

BRANCHES = ("environment", "reactant", "product")


def feature_groups(model, metadata, include_details=True):
    names = list(molecular_descriptor_names(metadata["global_descriptor_config"]["include_3d"]))
    if metadata["global_descriptor_config"]["per_molecule_descriptor_names"] != names:
        raise ValueError("Descriptor order differs from checkpoint")
    fp, d, two = model.fingerprint_nbits, len(names), len(FAST_2D_DESCRIPTOR_NAMES)
    env_fp = model.environment_gnn.global_layer.in_features - d - len(XTB_GLOBAL_FEATURE_NAMES)
    if env_fp != 256 or model.reactant_gnn.global_layer.in_features != 2*fp + 2*d + len(XTB_REACTION_GLOBAL_NAMES):
        raise ValueError("Unsupported global feature layout")
    groups = []
    def add(name, level, kind, branches=(), columns=None, strata=None):
        groups.append(dict(name=name, level=level, kind=kind, branches=list(branches),
                           columns=columns or {}, strata=strata or {}))
    for branch in BRANCHES:
        add(branch + " graph", "graph", "branch", [branch])
        add(branch + " local graph features", "local_graph", "local", [branch])
    add("All three graphs (joint)", "graph", "branch", BRANCHES)
    add("Temperature (physical equation)", "graph", "temperature")
    def cols(branches, start, end):
        return {b: list(range(start, end)) for b in branches}
    rp = ("reactant", "product")
    add("Solvent fingerprint", "feature_group", "global", columns=cols(["environment"], 0, env_fp))
    add("Ligand fingerprint (shared)", "feature_group", "global", columns=cols(rp, 0, fp))
    add("Reactant/product fingerprints (joint)", "feature_group", "global", columns=cols(rp, fp, 2*fp))
    roles = [("Solvent", ("environment",), env_fp), ("Ligand", rp, 2*fp),
             ("Reactant/product", rp, 2*fp+d)]
    for role, branches, start in roles:
        add(role + " 2D descriptors", "feature_group", "global", columns=cols(branches, start, start+two))
        if d > two:
            add(role + " 3D descriptors", "feature_group", "global", columns=cols(branches, start+two, start+d))
        if include_details:
            for i, name in enumerate(names):
                add(role + ": " + name, "descriptor", "global", columns=cols(branches, start+i, start+i+1))
    xtb_roles = [("Solvent", ("environment",), env_fp+d), ("Ligand", rp, 2*fp+2*d),
                 ("Reactant/product", rp, 2*fp+2*d+14), ("Boron", rp, 2*fp+2*d+28)]
    all_xtb = {}
    for role, branches, start in xtb_roles:
        block = cols(branches, start, start+14)
        add(role + " xTB (values + masks)", "feature_group", "global", columns=block)
        for branch, indices in block.items():
            all_xtb.setdefault(branch, []).extend(indices)
        if include_details:
            for i, name in enumerate(XTB_GLOBAL_FEATURE_NAMES[:10]):

                add(role + ": " + name, "descriptor", "global", columns=cols(branches, start+i, start+i+1),
                    strata=cols(branches, start+10, start+14))
    add("All global xTB blocks (joint)", "feature_group", "global", columns=all_xtb)
    return groups
