import dgl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, r2_score
from src.train_eval import collate_regression
from analysis.feature_importance.schema import BRANCHES


def metrics(truth, predicted):
    truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
    if truth.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Incomplete or nonfinite prediction")
    a, b = [np.searchsorted([23., 25.], values, side="right") for values in (truth, predicted)]
    return dict(mae=float(np.abs(truth-predicted).mean()), rmse=float(np.sqrt(np.mean((truth-predicted)**2))),
                r2=float(r2_score(truth, predicted)), accuracy=float(accuracy_score(a, b)),
                macro_f1=float(f1_score(a, b, labels=[0, 1, 2], average="macro", zero_division=0)))


class FrozenPredictor:


    def __init__(self, model, dataset, device, batch_size=4):
        if model.training or model.ablate_branches:
            raise ValueError("Requires eval-mode full three-branch model")
        self.model, self.device = model, device
        buffers = {key: [] for key in ["context", "temperature", "direct"] +
                   [f"{branch}_{part}" for branch in BRANCHES for part in ("local", "global")]}
        self.names = []
        with torch.no_grad():
            for offset in range(0, len(dataset), batch_size):
                batch = dataset[offset:offset+batch_size]
                names, *rest = collate_regression(batch, device)
                graphs, temperature = rest[:3], rest[3]
                self.names.extend(names)
                buffers["temperature"].append(temperature[:, None])
                buffers["direct"].append(torch.cat(model(*graphs, temperature, return_components=True), dim=1))
                for branch, graph in zip(BRANCHES, graphs):
                    layer = getattr(model, branch + "_gnn")
                    local = torch.relu(layer.gnn_layer(graph, graph.ndata["node"], graph.edata["edge"]))
                    with graph.local_scope():
                        graph.ndata["_importance_local"] = local
                        buffers[branch+"_local"].append(dgl.mean_nodes(graph, "_importance_local"))
                    globals_ = torch.stack([sample[1+BRANCHES.index(branch)].ndata["global"][0] for sample in batch]).to(device)

                    for sample in batch:
                        values = sample[1+BRANCHES.index(branch)].ndata["global"]
                        if not torch.equal(values, values[0:1].expand_as(values)):
                            raise ValueError("Nonconstant graph-global features cannot use this evaluator")
                    buffers[branch+"_global"].append(globals_)
                    if branch == "reactant":
                        buffers["context"].append(model.catalyst_context(graph, local))
        self.data = {name: torch.cat(parts) for name, parts in buffers.items()}
        with torch.no_grad():
            self.projected = {b: getattr(model, b+"_gnn").global_layer(self.data[b+"_global"]) for b in BRANCHES}
            self.baseline = self.predict()
        error = (self.baseline-self.data["direct"]).abs().max().item()
        if error > 2e-4:
            raise ValueError(f"Cached prediction differs from graph forward: max error {error}")
        np.testing.assert_array_equal(np.searchsorted([23., 25.], self.baseline[:, 0].cpu().numpy(), side="right"),
                                      np.searchsorted([23., 25.], self.data["direct"][:, 0].cpu().numpy(), side="right"))
        self.max_direct_error = error

    def permutation(self, group, seed):
        rng = np.random.RandomState(seed)
        n = len(self.names)
        order = np.arange(n)
        if group["strata"]:
            masks = torch.cat([self.data[b+"_global"][:, indices] for b, indices in group["strata"].items()], dim=1)
            _, labels = np.unique(masks.cpu().numpy(), axis=0, return_inverse=True)
            for label in np.unique(labels):
                indices = np.flatnonzero(labels == label)
                order[indices] = rng.permutation(indices)
        else:
            order = rng.permutation(n)
        return torch.as_tensor(order, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict(self, group=None, order=None):
        context, temperature = self.data["context"], self.data["temperature"]
        encoded = []
        for b in BRANCHES:
            local, projected = self.data[b+"_local"], self.projected[b]
            if group is not None:
                if group["kind"] in ("branch", "local") and b in group["branches"]:
                    local = local[order]
                    if b == "reactant":
                        context = context[order]
                    if group["kind"] == "branch":
                        projected = projected[order]
                if b in group["columns"]:
                    cols = group["columns"][b]
                    values = self.data[b+"_global"]
                    weight = getattr(self.model, b+"_gnn").global_layer.weight[:, cols]
                    projected = projected + (values[order][:, cols]-values[:, cols]) @ weight.T
            encoded.append(local + torch.relu(projected))
        if group is not None and group["kind"] == "temperature":
            temperature = temperature[order]
        hs = self.model.fc2(torch.relu(self.model.fc1(torch.cat([*encoded, context], dim=1))))
        h, s = hs[:, :1], hs[:, 1:2]
        return torch.cat([h-temperature*s*.001, h, s], dim=1)

    def varies(self, group):
        if group["kind"] == "global":
            tensors = [self.data[b+"_global"][:, cols] for b, cols in group["columns"].items()]
        elif group["kind"] == "temperature":
            tensors = [self.data["temperature"]]
        else:
            tensors = [self.data[b+"_local"] for b in group["branches"]]
            if group["kind"] == "branch":
                tensors.extend(self.data[b+"_global"] for b in group["branches"])
            if "reactant" in group["branches"]:
                tensors.append(self.data["context"])
        return any(not torch.equal(t, t[:1].expand_as(t)) for t in tensors)


def evaluate_group(predictor, group, truth, repeats, random_seed):
    baseline = predictor.baseline.cpu().numpy()
    original = metrics(truth, baseline[:, 0])
    varies = predictor.varies(group)
    rows = []
    for repeat in range(repeats):
        order = predictor.permutation(group, (random_seed + repeat) % (2**32))
        values = predictor.predict(group, order).cpu().numpy() if varies else baseline
        scores = metrics(truth, values[:, 0])
        rows.append(dict(feature=group["name"], level=group["level"], repeat=repeat, varies=varies,
                         mae_increase=scores["mae"]-original["mae"], rmse_increase=scores["rmse"]-original["rmse"],
                         r2_decrease=original["r2"]-scores["r2"], accuracy_drop_pp=100*(original["accuracy"]-scores["accuracy"]),
                         macro_f1_decrease=original["macro_f1"]-scores["macro_f1"],
                         mean_abs_g_change=float(np.abs(values[:, 0]-baseline[:, 0]).mean()),
                         mean_abs_h_change=float(np.abs(values[:, 1]-baseline[:, 1]).mean()),
                         mean_abs_ts_change=float(np.abs((values[:, 1]-values[:, 0])-(baseline[:, 1]-baseline[:, 0])).mean())))
    return rows


@torch.no_grad()
def verify_intervention(predictor, dataset, group, order, count=4):

    samples = []
    indices = order.cpu().tolist()
    for i in range(min(count, len(dataset))):
        original, donor = dataset[i], dataset[indices[i]]
        graphs = []
        for j, branch in enumerate(BRANCHES, start=1):
            replaced = group["kind"] in ("branch", "local") and branch in group["branches"]
            graph = (donor[j] if replaced else original[j]).clone()
            if replaced and group["kind"] == "local":
                graph.ndata["global"] = original[j].ndata["global"][:1].expand(graph.num_nodes(), -1).clone()
            if branch in group["columns"]:
                vector = original[j].ndata["global"][:1].clone()
                vector[:, group["columns"][branch]] = donor[j].ndata["global"][:1, group["columns"][branch]]
                graph.ndata["global"] = vector.expand(graph.num_nodes(), -1).clone()
            graphs.append(graph)
        temperature = donor[4] if group["kind"] == "temperature" else original[4]
        samples.append((original[0], *graphs, temperature, original[5]))
    _, a, b, c, temperature, _ = collate_regression(samples, predictor.device)
    direct = torch.cat(predictor.model(a, b, c, temperature, return_components=True), dim=1)
    cached = predictor.predict(group, order)[:len(samples)]
    torch.testing.assert_close(cached, direct, atol=2e-4, rtol=1e-5)
    return float((cached-direct).abs().max())
