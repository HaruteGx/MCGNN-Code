import inspect
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

from src.features import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES

BRANCHES = ("environment", "reactant", "product")


def summarize_graph(graph, branch):

    nodes = graph.ndata["node"].detach().cpu().double()
    edges = graph.edata["edge"].detach().cpu().double()
    if nodes.shape[1] != len(NODE_FEATURE_NAMES) or edges.shape[1] != len(EDGE_FEATURE_NAMES):
        raise ValueError("Graph feature dimensions differ from the current model schema")
    local = torch.cat([nodes.mean(0), edges.mean(0) if len(edges) else torch.zeros(edges.shape[1], dtype=torch.float64)]).numpy()
    local_names = ([f"{branch}.node_mean.{name}" for name in NODE_FEATURE_NAMES]
                   + [f"{branch}.edge_mean.{name}" for name in EDGE_FEATURE_NAMES])
    global_nodes = graph.ndata["global"].detach().cpu()
    if not torch.equal(global_nodes, global_nodes[0].expand_as(global_nodes)):
        raise ValueError("Expected the current model's repeated graph-level global vector")
    global_values = global_nodes[0].double().numpy()
    topology = np.array([graph.num_nodes(), graph.num_edges(), int((graph.in_degrees() == 0).sum())], dtype=float)
    names = local_names + [f"{branch}.global_{i}" for i in range(len(global_values))] + [
        f"{branch}.node_count", f"{branch}.directed_edge_count", f"{branch}.isolated_node_count"]
    return np.concatenate([local, global_values, topology]), names, local, local_names


def extract_vectors(model, dataset, batch_size=8):

    raw, local, latent, predictions = [], [], [], []
    schema = None
    for sample in dataset:
        if len(sample) != 6:
            raise ValueError("Current samples must include separate Kelvin temperature and target")
        rows = [summarize_graph(graph, branch) for graph, branch in zip(sample[1:4], BRANCHES)]
        current_schema = dict(graph_summary=sum([r[1] for r in rows], []), local=sum([r[3] for r in rows], []))
        if schema is not None and schema != current_schema:
            raise ValueError("Graph dimensions changed between records")
        schema = current_schema
        raw.append(np.concatenate([r[0] for r in rows]))
        local.append(np.concatenate([r[2] for r in rows]))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = dataset[start:start+batch_size]
            graphs = [dgl.batch([s[i] for s in samples]) for i in (1, 2, 3)]
            temperature = torch.tensor([s[4] for s in samples]).reshape(-1, 1)
            embedding = model.encode_reaction(*graphs)
            hs = model.fc2(embedding)
            reconstructed = hs[:, :1]-temperature*hs[:, 1:2]/1000
            direct = model(*graphs, temperature)
            torch.testing.assert_close(reconstructed, direct, rtol=1e-5, atol=2e-5)
            latent.append(embedding.cpu().numpy())
            predictions.extend(direct[:, 0].cpu().tolist())
    return dict(graph_summary=np.asarray(raw), local=np.asarray(local), model=np.vstack(latent),
                predicted_g=np.asarray(predictions), names=np.asarray([str(s[0]) for s in dataset])), schema


def project_space(matrix, seed=666, perplexity=30., iterations=2000, pca_components=50):
    matrix = np.asarray(matrix, float)
    if len(matrix) < 4 or not np.isfinite(matrix).all():
        raise ValueError("Need at least four finite sample vectors")
    active = np.ptp(matrix, axis=0) > 1e-10
    if active.sum() < 2:
        raise ValueError("Need at least two nonconstant features")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix[:, active])
    components = min(pca_components, len(matrix)-1, standardized.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    reduced = pca.fit_transform(standardized)
    effective_perplexity = min(perplexity, float(len(matrix)-1))
    kwargs = dict(n_components=2, perplexity=effective_perplexity, init="pca", learning_rate="auto",
                  metric="euclidean", random_state=seed, n_jobs=1)
    kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = iterations
    estimator = TSNE(**kwargs)
    coordinates = estimator.fit_transform(reduced)
    audit = dict(input_dimensions=matrix.shape[1], active_dimensions=int(active.sum()), pca_components=components,
                 pca_explained_variance=float(pca.explained_variance_ratio_.sum()), requested_perplexity=perplexity,
                 effective_perplexity=effective_perplexity, iterations=iterations, random_seed=seed,
                 metric="euclidean on standardized PCA vectors", kl_divergence=float(estimator.kl_divergence_),
                 preprocessing_fit="all displayed Ir features; descriptive visualization only; no targets used")
    return coordinates, reduced, audit


def nearest_neighbors(matrix):

    matrix = np.asarray(matrix, float)
    if len(matrix) < 2:
        raise ValueError("Need two rows for nearest neighbors")
    estimator = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=1).fit(matrix)
    nearest, distance = [], []
    for start in range(0, len(matrix), 128):
        distances, indices = estimator.kneighbors(matrix[start:start+128])
        for offset, (ds, ids) in enumerate(zip(distances, indices)):
            choice = np.flatnonzero(ids != start+offset)[0]
            nearest.append(ids[choice])
            distance.append(ds[choice])
    nearest = np.asarray(nearest, dtype=int)
    denominator = np.linalg.norm(matrix, axis=1)*np.linalg.norm(matrix[nearest], axis=1)
    cosine = np.divide((matrix*matrix[nearest]).sum(1), denominator,
                       out=np.full(len(matrix), np.nan), where=denominator > 1e-12)
    return nearest, np.asarray(distance), np.clip(cosine, -1, 1)


def vif_table(matrix, names):

    matrix = np.asarray(matrix, float)
    active = np.ptp(matrix, axis=0) > 1e-10
    scaled = StandardScaler().fit_transform(matrix[:, active])
    result, position = [], 0
    for index, name in enumerate(names):
        if not active[index]:
            result.append(dict(feature=name, VIF=np.nan, status="constant"))
            continue
        target, others = scaled[:, position], np.delete(scaled, position, axis=1)
        if others.shape[1] == 0:
            score = 0.
        else:
            score = LinearRegression().fit(others, target).score(others, target)
        value = np.inf if 1-score < 1e-10 else 1/max(1-score, 1e-10)
        result.append(dict(feature=name, VIF=value, status="collinear" if np.isinf(value) else "finite"))
        position += 1
    return pd.DataFrame(result).sort_values("VIF", ascending=False, na_position="last")
