import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from .features import (EDGE_FEATURE_DIM, NODE_FEATURE_DIM, NODE_FEATURE_NAMES,
                       EDGE_FEATURE_NAMES, METAL_NODE_FEATURE_INDEX,
                       DONOR_NODE_FEATURE_INDEX, REACTION_SITE_NODE_FEATURE_INDEX)
from .xtb_features import xtb_feature_config, validate_xtb_config

class EnvironmentGNN(nn.Module):
    def __init__(self, global_feature_dim=256, dropout=0.08):
        super().__init__()
        in_feats_node = NODE_FEATURE_DIM
        in_feats_edge = EDGE_FEATURE_DIM
        hidden_feats = 256
        out_feats = 128

        self.edge_net = nn.Sequential(
            nn.Linear(in_feats_edge, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, in_feats_node * out_feats)
        )
        self.gnn_layer = dglnn.NNConv(in_feats_node, out_feats, self.edge_net, aggregator_type='mean')
        self.global_layer = nn.Linear(global_feature_dim, out_feats)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, env_graph):
        node_features = env_graph.ndata['node']
        edge_features = env_graph.edata['edge']
        global_features = env_graph.ndata['global']  

        node_out = F.relu(self.gnn_layer(env_graph, node_features, edge_features))
        node_out = self.dropout(node_out)

        global_out = F.relu(self.global_layer(global_features))
        global_out = self.dropout(global_out)

        combined = node_out + global_out
        with env_graph.local_scope():
            env_graph.ndata['combined_out'] = combined
            return dgl.mean_nodes(env_graph, 'combined_out')  

class ReactantGNN(nn.Module):
    def __init__(self, global_feature_dim, hidden_feats=2048, out_feats=1024, dropout=0.08):
        super().__init__()
        in_feats_node = NODE_FEATURE_DIM
        in_feats_edge = EDGE_FEATURE_DIM


        self.out_feats = out_feats

        self.edge_net = nn.Sequential(
            nn.Linear(in_feats_edge, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, in_feats_node * out_feats)
        )
        self.gnn_layer = dglnn.NNConv(
            in_feats_node, out_feats, self.edge_net, aggregator_type='mean', residual=True)
        self.global_layer = nn.Linear(global_feature_dim, out_feats)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, return_node_states=False):
        node_features = g.ndata['node']
        edge_features = g.edata['edge']
        global_features = g.ndata['global']  

        node_out = F.relu(self.gnn_layer(g, node_features, edge_features))
        node_out = self.dropout(node_out)

        global_out = F.relu(self.global_layer(global_features))
        global_out = self.dropout(global_out)

        combined = node_out + global_out
        with g.local_scope():
            g.ndata['combined_out'] = combined
            pooled = dgl.mean_nodes(g, 'combined_out')
        return (pooled, node_out) if return_node_states else pooled

class ProductGNN(nn.Module):
    def __init__(self, global_feature_dim, hidden_feats=2048, out_feats=1024, dropout=0.08):
        super().__init__()
        in_feats_node = NODE_FEATURE_DIM
        in_feats_edge = EDGE_FEATURE_DIM


        self.out_feats = out_feats

        self.edge_net = nn.Sequential(
            nn.Linear(in_feats_edge, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, in_feats_node * out_feats)
        )
        self.gnn_layer = dglnn.NNConv(in_feats_node, out_feats, self.edge_net, aggregator_type='mean')
        self.global_layer = nn.Linear(global_feature_dim, out_feats)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g):
        node_features = g.ndata['node']
        edge_features = g.edata['edge']
        global_features = g.ndata['global']

        node_out = F.relu(self.gnn_layer(g, node_features, edge_features))
        node_out = self.dropout(node_out)

        global_out = F.relu(self.global_layer(global_features))
        global_out = self.dropout(global_out)

        combined = node_out + global_out
        with g.local_scope():
            g.ndata['combined_out'] = combined
            return dgl.mean_nodes(g, 'combined_out')  

class CatalystContext(nn.Module):


    def __init__(self, node_hidden_dim, context_dim=64, dropout=0.2):
        super().__init__()
        scale = torch.ones(NODE_FEATURE_DIM)
        scale[:17] = torch.tensor([100, 4, 6, 1, 1, 1, 4, 4, 1, 3, 3, 2, 4, 15, 10, 1, 1])
        self.register_buffer("raw_feature_scale", scale)
        self.metal_projection = nn.Sequential(nn.Linear(NODE_FEATURE_DIM, context_dim), nn.SiLU())
        self.node_projection = nn.Sequential(nn.Linear(node_hidden_dim, context_dim), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(context_dim * 6, context_dim), nn.SiLU(), nn.Dropout(dropout))

    @staticmethod
    def masked_mean(graph, values, mask):
        with graph.local_scope():
            graph.ndata['_context_values'] = values * mask
            graph.ndata['_context_mask'] = mask
            return (dgl.sum_nodes(graph, '_context_values') /
                    dgl.sum_nodes(graph, '_context_mask').clamp_min(1.0))

    def forward(self, graph, node_states):
        raw = graph.ndata['node']
        metal_mask = raw[:, METAL_NODE_FEATURE_INDEX:METAL_NODE_FEATURE_INDEX+1]
        donor_mask = raw[:, DONOR_NODE_FEATURE_INDEX:DONOR_NODE_FEATURE_INDEX+1]
        site_mask = raw[:, REACTION_SITE_NODE_FEATURE_INDEX:REACTION_SITE_NODE_FEATURE_INDEX+1]
        metal = self.metal_projection(self.masked_mean(graph, raw / self.raw_feature_scale, metal_mask))
        projected = self.node_projection(node_states)
        neighborhood = self.masked_mean(graph, projected, metal_mask)
        donors = self.masked_mean(graph, projected, donor_mask)
        sites = self.masked_mean(graph, projected, site_mask)
        return self.fusion(torch.cat([metal, neighborhood, donors, sites,
                                      metal * donors, metal * sites], dim=-1))


class ReactionEnergyPredictionModel(nn.Module):


    ENTROPY_CAL_TO_KCAL = 1.0e-3
    ABLATABLE_GRAPH_BRANCHES = ("environment", "reactant", "product")

    def __init__(self, global_feature_dim,
                 environment_global_feature_dim=256,
                 dropout=0.08,
                 reactant_hidden_feats=2048,
                 reactant_out_feats=1024,
                 product_hidden_feats=2048,
                 product_out_feats=1024,
                 fc_hidden_feats=256,
                 catalyst_context_dim=64,
                 ablate_branches=()):
        super().__init__()
        ablate_branches = tuple(sorted(set(ablate_branches)))
        unknown_branches = set(ablate_branches).difference(
            self.ABLATABLE_GRAPH_BRANCHES
        )
        if unknown_branches:
            raise ValueError(
                "Unknown graph branches to ablate: "
                f"{sorted(unknown_branches)}. Choose from "
                f"{self.ABLATABLE_GRAPH_BRANCHES}."
            )
        self.ablate_branches = frozenset(ablate_branches)
        self.environment_gnn = EnvironmentGNN(
            global_feature_dim=environment_global_feature_dim, dropout=dropout
        )
        self.reactant_gnn = ReactantGNN(
            global_feature_dim,
            hidden_feats=reactant_hidden_feats,
            out_feats=reactant_out_feats,
            dropout=dropout,
        )
        self.product_gnn = ProductGNN(
            global_feature_dim,
            hidden_feats=product_hidden_feats,
            out_feats=product_out_feats,
            dropout=dropout,
        )


        self.catalyst_context = CatalystContext(reactant_out_feats, catalyst_context_dim, dropout)
        self.xtb_feature_config = xtb_feature_config()
        self.fc1 = nn.Linear(128 + reactant_out_feats + product_out_feats + catalyst_context_dim, fc_hidden_feats)
        self.fc2 = nn.Linear(fc_hidden_feats, 2)

    def encode_reaction(self, env_graph, reactant_graph, product_graph):
        env_f = self.environment_gnn(env_graph)
        rea_f, rea_nodes = self.reactant_gnn(reactant_graph, return_node_states=True)
        pro_f = self.product_gnn(product_graph)
        metal_f = self.catalyst_context(reactant_graph, rea_nodes)


        if "environment" in self.ablate_branches:
            env_f = torch.zeros_like(env_f)
        if "reactant" in self.ablate_branches:
            rea_f = torch.zeros_like(rea_f)
            metal_f = torch.zeros_like(metal_f)
        if "product" in self.ablate_branches:
            pro_f = torch.zeros_like(pro_f)
        x = torch.cat([env_f, rea_f, pro_f, metal_f], dim=-1)
        return F.relu(self.fc1(x))

    def predict_thermodynamic_components(
        self, env_graph, reactant_graph, product_graph, temperature_kelvin
    ):

        latent = self.fc2(
            self.encode_reaction(env_graph, reactant_graph, product_graph)
        )
        delta_h_kcal_mol = latent[:, 0:1]
        delta_s_cal_mol_k = latent[:, 1:2]
        temperature_kelvin = torch.as_tensor(
            temperature_kelvin,
            dtype=latent.dtype,
            device=latent.device,
        ).reshape(-1, 1)
        if temperature_kelvin.shape[0] != latent.shape[0]:
            raise ValueError(
                "Temperature batch size does not match the graph batch: "
                f"{temperature_kelvin.shape[0]} vs {latent.shape[0]}."
            )
        if not torch.isfinite(temperature_kelvin).all() or torch.any(
            temperature_kelvin <= 0
        ):
            raise ValueError("All temperatures must be finite and greater than 0 K.")
        delta_g_kcal_mol = delta_h_kcal_mol - (
            temperature_kelvin
            * delta_s_cal_mol_k
            * self.ENTROPY_CAL_TO_KCAL
        )
        return delta_g_kcal_mol, delta_h_kcal_mol, delta_s_cal_mol_k

    def forward(
        self,
        env_graph,
        reactant_graph,
        product_graph,
        temperature_kelvin,
        return_components=False,
    ):
        outputs = self.predict_thermodynamic_components(
            env_graph, reactant_graph, product_graph, temperature_kelvin
        )
        return outputs if return_components else outputs[0]


def regression_model_config_from_checkpoint(state_dict, metadata=None):

    if int(state_dict["fc2.weight"].shape[0]) != 2:
        raise ValueError(
            "Regression checkpoint must have the two-value Delta-H/Delta-S "
            "head. Retrain legacy direct-Delta-G checkpoints with "
            "scripts/train_full_model.py."
        )
    if "catalyst_context.metal_projection.0.weight" not in state_dict:
        raise ValueError("Legacy checkpoint has no xTB/metal context branch. Run scripts/train_full_model.py to retrain.")
    metadata = metadata or {}
    saved_config = metadata.get("model_config", {}) or {}
    return {
        "global_feature_dim": int(
            state_dict["reactant_gnn.global_layer.weight"].shape[1]
        ),
        "environment_global_feature_dim": int(
            state_dict["environment_gnn.global_layer.weight"].shape[1]
        ),
        "dropout": float(saved_config.get("dropout", 0.08)),
        "reactant_hidden_feats": int(
            state_dict["reactant_gnn.edge_net.0.weight"].shape[0]
        ),
        "reactant_out_feats": int(
            state_dict["reactant_gnn.global_layer.weight"].shape[0]
        ),
        "product_hidden_feats": int(
            state_dict["product_gnn.edge_net.0.weight"].shape[0]
        ),
        "product_out_feats": int(
            state_dict["product_gnn.global_layer.weight"].shape[0]
        ),
        "fc_hidden_feats": int(state_dict["fc1.weight"].shape[0]),
        "catalyst_context_dim": int(state_dict["catalyst_context.metal_projection.0.weight"].shape[0]),
        "ablate_branches": tuple(saved_config.get("ablate_branches", ())),
    }


def build_regression_model_from_checkpoint(state_dict, metadata=None):

    config = regression_model_config_from_checkpoint(state_dict, metadata)
    metadata = metadata or {}
    validate_xtb_config(metadata.get("xtb_feature_config"))
    if (metadata.get("node_feature_schema") != list(NODE_FEATURE_NAMES)
            or metadata.get("edge_feature_schema") != list(EDGE_FEATURE_NAMES)):
        raise ValueError("Checkpoint node/edge semantics differ from the current xTB schema")
    for branch in ("environment_gnn", "reactant_gnn", "product_gnn"):
        output_dim = int(state_dict[f"{branch}.global_layer.weight"].shape[0])
        saved_projection_dim = int(
            state_dict[f"{branch}.edge_net.2.weight"].shape[0]
        )
        if saved_projection_dim != NODE_FEATURE_DIM * output_dim:
            raise ValueError(
                "Checkpoint uses an incompatible node feature schema. Retrain "
                "the Regress Test model with the current feature code."
            )
        saved_edge_dim = int(
            state_dict[f"{branch}.edge_net.0.weight"].shape[1]
        )
        if saved_edge_dim != EDGE_FEATURE_DIM:
            raise ValueError(
                "Checkpoint uses an incompatible edge feature schema. Retrain "
                "the Regress Test model with the current feature code."
            )
    model = ReactionEnergyPredictionModel(**config)
    metadata = metadata or {}
    fingerprint_config = metadata.get("fingerprint_config", {}) or {}
    model.fingerprint_nbits = int(
        fingerprint_config.get(
            "n_bits_per_fingerprint", config["global_feature_dim"] // 2
        )
    )
    model.global_descriptor_config = dict(
        metadata.get("global_descriptor_config", {}) or {}
    )
    model.xtb_feature_config = dict(metadata["xtb_feature_config"])
    model.checkpoint_metadata = metadata
    return model, config
