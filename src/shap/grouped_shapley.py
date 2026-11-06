import numpy as np
import torch


def estimate_shapley(game, n_features, repeats=16, seed=42):


    if n_features < 1 or repeats < 2:
        raise ValueError("Require features and at least two independent order pairs")
    rng = np.random.default_rng(seed)
    estimates = []
    endpoints = np.asarray(game(np.stack([np.zeros(n_features), np.ones(n_features)])), float)
    for _ in range(repeats):
        order = rng.permutation(n_features)
        pair = []
        for sequence in (order, order[::-1]):
            masks = np.zeros((n_features + 1, n_features), dtype=np.float32)
            masks[1:] = np.cumsum(np.eye(n_features, dtype=np.float32)[sequence], axis=0)
            values = np.asarray(game(masks), float)
            if values.shape != (n_features + 1,) or not np.isfinite(values).all():
                raise ValueError("Invalid coalition output")
            attribution = np.empty(n_features)
            attribution[sequence] = np.diff(values)
            pair.append(attribution)
        estimates.append(np.mean(pair, axis=0))
    estimates = np.asarray(estimates)
    phi = estimates.mean(axis=0)
    return dict(values=phi, sem=estimates.std(axis=0, ddof=1)/np.sqrt(repeats),
                repeats=estimates, base=float(endpoints[0]), prediction=float(endpoints[1]),
                residual=float(phi.sum() + endpoints[0] - endpoints[1]))


def disjoint_groups(model, metadata, level="descriptors"):

    from analysis.feature_importance.schema import feature_groups, BRANCHES
    all_groups = feature_groups(model, metadata, True)
    if level == "groups":
        groups = [g for g in all_groups if g["level"] == "feature_group"
                  and g["name"] != "All global xTB blocks (joint)"]
    elif level == "descriptors":
        groups = [g for g in all_groups if g["level"] == "descriptor"
                  or (g["level"] == "feature_group" and "fingerprint" in g["name"].lower())]

        for g in all_groups:
            if g["level"] == "feature_group" and "(values + masks)" in g["name"]:
                groups.append(dict(name=g["name"].replace("(values + masks)", "availability masks"),
                                   kind="global", columns={b: c[-4:] for b, c in g["columns"].items()}))
    else:
        raise ValueError("Unknown feature resolution")
    groups = [dict(name=g["name"], kind="global", columns=g["columns"]) for g in groups]
    for b in BRANCHES:
        groups.append(dict(name=b + " local graph" + (" + catalyst context" if b == "reactant" else ""),
                           kind="local", branch=b))
    for b in BRANCHES:
        columns = [c for g in groups for c in g.get("columns", {}).get(b, [])]
        expected = getattr(model, b+"_gnn").global_layer.in_features
        if sorted(columns) != list(range(expected)):
            raise ValueError(f"Groups overlap or omit global fields in {b}")
    return groups


def attribution_partition(groups, fingerprint_mode="include"):

    if fingerprint_mode not in ("include", "fixed"):
        raise ValueError("fingerprint_mode must be include or fixed")
    fixed = [g for g in groups if fingerprint_mode == "fixed"
             and g["kind"] == "global" and "fingerprint" in g["name"].lower()]
    active = [g for g in groups if g not in fixed]
    if not active:
        raise ValueError("No SHAP players remain")
    return active, fixed


class CoalitionGame:


    @torch.no_grad()
    def __init__(self, predictor, groups, target, background, fixed_groups=()):
        import torch
        from analysis.feature_importance.schema import BRANCHES
        self.torch, self.predictor, self.groups = torch, predictor, groups
        self.target, self.background = target, list(background)
        self.fixed_groups = list(fixed_groups)
        occupied = {(b, c) for g in groups for b, cols in g.get("columns", {}).items() for c in cols}
        for g in self.fixed_groups:
            if g["kind"] != "global":
                raise ValueError("Only global groups may be fixed explicitly")
            for b, cols in g["columns"].items():
                for c in cols:
                    if (b, c) in occupied:
                        raise ValueError("Fixed and active groups must not overlap")
                    occupied.add((b, c))
        self.branches = BRANCHES
        self.parts, self.slices = {}, {}
        offset = 0
        for b in BRANCHES:
            for role, values in (("local", predictor.data[b+"_local"]),
                                 ("projected", predictor.projected[b])):
                self.parts[b+"_"+role] = values
        self.parts.update(context=predictor.data["context"], temperature=predictor.data["temperature"])
        for name, values in self.parts.items():
            self.slices[name] = slice(offset, offset+values.shape[1])
            offset += values.shape[1]
        self.base = torch.cat([v[self.background] for v in self.parts.values()], dim=1)


        self.base[:, self.slices["temperature"]] = predictor.data["temperature"][target]


        for group in self.fixed_groups:
            for b, cols in group["columns"].items():
                values = predictor.data[b+"_global"]
                weight = getattr(predictor.model, b+"_gnn").global_layer.weight[:, cols]
                self.base[:, self.slices[b+"_projected"]] += (values[target, cols]-values[self.background][:, cols]) @ weight.T
        self.delta = torch.zeros((len(groups), len(background), offset), device=predictor.device)
        for j, group in enumerate(groups):
            if group["kind"] == "global":
                for b, cols in group["columns"].items():
                    values = predictor.data[b+"_global"]
                    weight = getattr(predictor.model, b+"_gnn").global_layer.weight[:, cols]
                    self.delta[j, :, self.slices[b+"_projected"]] = (values[target, cols]-values[self.background][:, cols]) @ weight.T
            else:
                names = [group["branch"]+"_local"]
                if group.get("branch") == "reactant":
                    names.append("context")
                for name in names:
                    self.delta[j, :, self.slices[name]] = self.parts[name][target]-self.parts[name][self.background]

    @torch.no_grad()
    def reference_value(self):

        p = self.predictor
        encoded = []
        for b in self.branches:
            values = p.data[b+"_global"][self.background].clone()
            for group in self.fixed_groups:
                if b in group["columns"]:
                    cols = group["columns"][b]
                    values[:, cols] = p.data[b+"_global"][self.target, cols]
            projected = getattr(p.model, b+"_gnn").global_layer(values)
            encoded.append(p.data[b+"_local"][self.background] + torch.relu(projected))
        hs = p.model.fc2(torch.relu(p.model.fc1(torch.cat([*encoded, p.data["context"][self.background]], dim=1))))
        t = p.data["temperature"][self.target, 0]
        return float((hs[:,0]-t*hs[:,1]*.001).mean())

    def __call__(self, masks):
        torch = self.torch
        masks = np.asarray(masks)
        if masks.ndim != 2 or masks.shape[1] != len(self.groups) or not np.isin(masks, [0, 1]).all():
            raise ValueError("Expected binary coalition masks")
        results = []
        with torch.no_grad():
            for start in range(0, len(masks), 128):
                m = torch.as_tensor(masks[start:start+128], dtype=self.base.dtype, device=self.base.device)
                state = self.base[None] + (m @ self.delta.reshape(len(self.groups), -1)).reshape(len(m), *self.base.shape)
                encoded = [state[..., self.slices[b+"_local"]] + torch.relu(state[..., self.slices[b+"_projected"]]) for b in self.branches]
                hs = self.predictor.model.fc2(torch.relu(self.predictor.model.fc1(torch.cat([*encoded, state[..., self.slices["context"]]], dim=-1))))
                value = hs[..., 0] - state[..., self.slices["temperature"]][..., 0]*hs[..., 1]*.001
                results.extend(value.mean(dim=1).cpu().numpy())
        return np.asarray(results)

    def verify(self, dataset, seed=0, trials=3):

        import torch
        from src.train_eval import collate_regression
        rng = np.random.default_rng(seed)
        worst = 0.
        masks = [np.zeros(len(self.groups), dtype=int), np.ones(len(self.groups), dtype=int)]
        masks += [rng.integers(0, 2, len(self.groups)) for _ in range(trials)]
        for mask in masks:
            original = dataset[self.target]
            samples = []
            for donor_index in self.background:
                donor = dataset[donor_index]
                graphs = []
                for j, b in enumerate(self.branches, 1):
                    local_idx = next(i for i, g in enumerate(self.groups) if g.get("branch") == b)
                    graph = (original[j] if mask[local_idx] else donor[j]).clone()
                    vector = donor[j].ndata["global"][:1].clone()
                    for group in self.fixed_groups:
                        if b in group["columns"]:
                            cols = group["columns"][b]
                            vector[:, cols] = original[j].ndata["global"][:1, cols]
                    for active, group in zip(mask, self.groups):
                        if active and b in group.get("columns", {}):
                            cols = group["columns"][b]
                            vector[:, cols] = original[j].ndata["global"][:1, cols]
                    graph.ndata["global"] = vector.expand(graph.num_nodes(), -1).clone()
                    graphs.append(graph)
                samples.append((original[0], *graphs, original[4], original[5]))
            with torch.no_grad():
                _, a, b, c, t, _ = collate_regression(samples, self.predictor.device)
                direct = self.predictor.model(a, b, c, t).mean().item()
            error = abs(float(self(mask[None])[0])-direct)
            worst = max(worst, error)
            if error > 2e-4:
                raise ValueError(f"Hybrid graph evaluation mismatch: {error}")
        return worst
