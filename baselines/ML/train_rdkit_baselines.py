from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.descriptors import molecular_descriptor_names, molecular_descriptor_vector
from src.features import morgan_fp_bits
from src.regression_classification import export_classification


rdBase.DisableLog("rdApp.warning")


TARGET_COLUMN = "G (kcal/mol)"
NAME_COLUMN = "Name"
N_SPLITS = 5
NUM_STRATIFICATION_BINS = 10
REACTION_FINGERPRINT_BITS = 256
ENVIRONMENT_FINGERPRINT_BITS = 128
SELECTED_FEATURES = 256
ENSEMBLE_TREES = 160
BOOSTING_STEPS = 120
QUICK_ENSEMBLE_TREES = 32
QUICK_BOOSTING_STEPS = 32
INCLUDE_3D_DESCRIPTORS = True
DESCRIPTOR_NUM_CONFORMERS = 3
DESCRIPTOR_CONFORMER_SEED = 20260822
OPTIMIZE_CONFORMERS = True
FEATURE_VERSION = "compact_rdkit_global_blocks_v3"

ROLE_COLUMNS = OrderedDict(
    (
        ("substrate", "sub-type"),
        ("product", "P"),
        ("ligand", "L-type"),
        ("boron", "B-type"),
        ("solvent", "sol"),
    )
)
REACTION_GROUP_COLUMNS = ("Cat", "sub-type", "L-type", "B-type", "P")
FORBIDDEN_MODEL_COLUMNS = (
    TARGET_COLUMN,
    "T / C",
    "conv./%",
    "Covn(r)",
    "Ratio%",
    "t / h",
    "[sol](ml)",
    "sub(mmol)",
    "[Cat](mol%)",
    "[L](mol%)",
    "B(eq)",
    "bpin_index_sub",
    "bpin_index Production",
)

MODEL_ORDER = (
    "DummyMean",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SGDRegressor",
    "BayesianRidge",
    "PLSRegression",
    "KNeighbors",
    "LinearSVR",
    "RBF_SVR",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "AdaBoost",
)


@lru_cache(maxsize=65536)
def _molecule_features(
    canonical_smiles: str,
    fingerprint_bits: int,
    include_3d_descriptors: bool,
    descriptor_num_conformers: int,
    descriptor_conformer_seed: int,
    optimize_conformers: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:

    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical SMILES: {canonical_smiles!r}")
    fingerprint = morgan_fp_bits(
        molecule,
        radius=2,
        nBits=int(fingerprint_bits),
    )
    descriptors = molecular_descriptor_vector(
        canonical_smiles,
        include_3d=bool(include_3d_descriptors),
        max_conformers=int(descriptor_num_conformers),
        conformer_seed=int(descriptor_conformer_seed),
        optimize_conformers=bool(optimize_conformers),
    )
    if descriptors is None:
        raise ValueError(f"RDKit descriptors failed for {canonical_smiles!r}")
    return (
        tuple(float(value) for value in np.asarray(fingerprint, dtype=np.float32)),
        tuple(float(value) for value in np.asarray(descriptors, dtype=np.float32)),
    )


def _canonical_smiles(value: object, *, role: str, sample_name: str) -> str:
    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"Missing {role} SMILES for sample {sample_name!r}.")
    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None:
        raise ValueError(f"Invalid {role} SMILES for sample {sample_name!r}: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _combined_reactant_smiles(substrate: str, boron: str, ligand: str) -> str:
    molecules = [Chem.MolFromSmiles(value) for value in (substrate, boron, ligand)]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Cannot combine invalid reactant component SMILES.")
    combined = molecules[0]
    for molecule in molecules[1:]:
        combined = Chem.CombineMols(combined, molecule)
    return Chem.MolToSmiles(combined, canonical=True, isomericSmiles=True)


def feature_names(
    fingerprint_bits: int = REACTION_FINGERPRINT_BITS,
    environment_fingerprint_bits: int = ENVIRONMENT_FINGERPRINT_BITS,
    include_3d_descriptors: bool = INCLUDE_3D_DESCRIPTORS,
) -> list[str]:
    descriptor_names = list(
        molecular_descriptor_names(include_3d=bool(include_3d_descriptors))
    )

    def fingerprint_names(branch: str, molecule: str, bits: int) -> list[str]:
        return [
            f"{branch}__{molecule}__morgan_r2_bit_{index:04d}"
            for index in range(int(bits))
        ]

    def descriptor_block(branch: str, molecule: str) -> list[str]:
        return [f"{branch}__{molecule}__{name}" for name in descriptor_names]


    return (
        fingerprint_names("environment", "solvent", environment_fingerprint_bits)
        + descriptor_block("environment", "solvent")
        + fingerprint_names("reactant", "ligand", fingerprint_bits)
        + fingerprint_names("reactant", "combined_molecule", fingerprint_bits)
        + descriptor_block("reactant", "ligand")
        + descriptor_block("reactant", "combined_molecule")
        + fingerprint_names("product", "ligand", fingerprint_bits)
        + fingerprint_names("product", "molecule", fingerprint_bits)
        + descriptor_block("product", "ligand")
        + descriptor_block("product", "molecule")
    )


def featurize_frame(
    frame: pd.DataFrame,
    fingerprint_bits: int = REACTION_FINGERPRINT_BITS,
    environment_fingerprint_bits: int = ENVIRONMENT_FINGERPRINT_BITS,
    include_3d_descriptors: bool = INCLUDE_3D_DESCRIPTORS,
    descriptor_num_conformers: int = DESCRIPTOR_NUM_CONFORMERS,
    descriptor_conformer_seed: int = DESCRIPTOR_CONFORMER_SEED,
    optimize_conformers: bool = OPTIMIZE_CONFORMERS,
) -> tuple[np.ndarray, list[str]]:

    missing = [column for column in (NAME_COLUMN, *ROLE_COLUMNS.values()) if column not in frame]
    if missing:
        raise KeyError(f"RDKit baseline table is missing columns: {missing}")
    rows: list[np.ndarray] = []
    total_rows = len(frame)
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        sample_name = str(row[NAME_COLUMN])
        canonical = {
            role: _canonical_smiles(row[column], role=role, sample_name=sample_name)
            for role, column in ROLE_COLUMNS.items()
        }
        reactant = _combined_reactant_smiles(
            canonical["substrate"], canonical["boron"], canonical["ligand"]
        )
        settings = (
            bool(include_3d_descriptors),
            int(descriptor_num_conformers),
            int(descriptor_conformer_seed),
            bool(optimize_conformers),
        )
        solvent_fp, solvent_desc = _molecule_features(
            canonical["solvent"], int(environment_fingerprint_bits), *settings
        )
        ligand_fp, ligand_desc = _molecule_features(
            canonical["ligand"], int(fingerprint_bits), *settings
        )
        reactant_fp, reactant_desc = _molecule_features(
            reactant, int(fingerprint_bits), *settings
        )
        product_fp, product_desc = _molecule_features(
            canonical["product"], int(fingerprint_bits), *settings
        )
        rows.append(
            np.concatenate(
                [
                    solvent_fp,
                    solvent_desc,
                    ligand_fp,
                    reactant_fp,
                    ligand_desc,
                    reactant_desc,
                    ligand_fp,
                    product_fp,
                    ligand_desc,
                    product_desc,
                ]
            ).astype(np.float32, copy=False)
        )
        if position % 100 == 0 or position == total_rows:
            print(f"  RDKit feature rows: {position}/{total_rows}", flush=True)
    matrix = np.asarray(rows, dtype=np.float32)
    names = feature_names(
        fingerprint_bits,
        environment_fingerprint_bits,
        include_3d_descriptors,
    )
    if matrix.shape != (len(frame), len(names)):
        raise RuntimeError(
            f"Feature shape mismatch: matrix={matrix.shape}, names={len(names)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("RDKit feature matrix contains non-finite values.")
    return matrix, names


class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):


    def __init__(self, k: int = 512):
        self.k = int(k)

    def fit(self, features: np.ndarray, targets: np.ndarray):
        self.selector_ = SelectKBest(f_regression, k=min(self.k, int(features.shape[1])))
        self.selector_.fit(features, targets)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.selector_.transform(features)

    def get_support(self, indices: bool = False):
        return self.selector_.get_support(indices=indices)


def _scaled_model(
    estimator: BaseEstimator, selected_features: int = SELECTED_FEATURES
) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", AdaptiveSelectKBest(selected_features)),
            ("scale", StandardScaler()),
            ("regressor", estimator),
        ]
    )


def _tree_model(
    estimator: BaseEstimator, selected_features: int = SELECTED_FEATURES
) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", AdaptiveSelectKBest(selected_features)),
            ("regressor", estimator),
        ]
    )


def build_model_registry(seed: int = 10, n_jobs: int = -1, quick: bool = False):

    trees = QUICK_ENSEMBLE_TREES if quick else ENSEMBLE_TREES
    boosting_steps = QUICK_BOOSTING_STEPS if quick else BOOSTING_STEPS
    registry = OrderedDict(
        (
            ("DummyMean", Pipeline([("impute", SimpleImputer()), ("regressor", DummyRegressor())])),
            ("LinearRegression", _scaled_model(LinearRegression(n_jobs=n_jobs))),
            ("Ridge", _scaled_model(Ridge(alpha=10.0))),
            ("Lasso", _scaled_model(Lasso(alpha=0.005, max_iter=20000, selection="cyclic"))),
            (
                "ElasticNet",
                _scaled_model(
                    ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=20000, selection="cyclic")
                ),
            ),
            (
                "SGDRegressor",
                _scaled_model(
                    SGDRegressor(
                        loss="huber",
                        penalty="elasticnet",
                        alpha=1e-4,
                        l1_ratio=0.15,
                        max_iter=5000,
                        tol=1e-4,
                        random_state=seed,
                    )
                ),
            ),
            ("BayesianRidge", _scaled_model(BayesianRidge(max_iter=500), selected_features=128)),
            ("PLSRegression", _scaled_model(PLSRegression(n_components=8, scale=False), 128)),
            (
                "KNeighbors",
                _scaled_model(KNeighborsRegressor(n_neighbors=7, weights="distance", p=2, n_jobs=n_jobs), 128),
            ),
            (
                "LinearSVR",
                _scaled_model(
                    LinearSVR(
                        C=1.0,
                        epsilon=0.1,
                        loss="squared_epsilon_insensitive",
                        dual=True,
                        max_iter=30000,
                        random_state=seed,
                    )
                ),
            ),
            ("RBF_SVR", _scaled_model(SVR(C=10.0, epsilon=0.1, gamma="scale"), 256)),
            (
                "DecisionTree",
                _tree_model(
                    DecisionTreeRegressor(
                        max_depth=8,
                        max_leaf_nodes=48,
                        min_samples_leaf=4,
                        random_state=seed,
                    )
                ),
            ),
            (
                "RandomForest",
                _tree_model(
                    RandomForestRegressor(
                        n_estimators=trees,
                        max_depth=10,
                        max_leaf_nodes=64,
                        max_features=0.35,
                        min_samples_leaf=2,
                        n_jobs=n_jobs,
                        random_state=seed,
                    )
                ),
            ),
            (
                "ExtraTrees",
                _tree_model(
                    ExtraTreesRegressor(
                        n_estimators=trees,
                        max_depth=10,
                        max_leaf_nodes=64,
                        max_features=0.5,
                        min_samples_leaf=2,
                        n_jobs=n_jobs,
                        random_state=seed,
                    )
                ),
            ),
            (
                "GradientBoosting",
                _tree_model(
                    GradientBoostingRegressor(
                        n_estimators=boosting_steps,
                        learning_rate=0.03 if not quick else 0.05,
                        max_depth=2,
                        min_samples_leaf=4,
                        loss="huber",
                        random_state=seed,
                    )
                ),
            ),
            (
                "HistGradientBoosting",
                _tree_model(
                    HistGradientBoostingRegressor(
                        max_iter=boosting_steps,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        min_samples_leaf=10,
                        l2_regularization=0.1,
                        random_state=seed,
                    )
                ),
            ),
            (
                "AdaBoost",
                _tree_model(
                    AdaBoostRegressor(
                        estimator=DecisionTreeRegressor(
                            max_depth=2,
                            max_leaf_nodes=12,
                            min_samples_leaf=4,
                            random_state=seed,
                        ),
                        n_estimators=boosting_steps,
                        learning_rate=0.03 if not quick else 0.05,
                        loss="square",
                        random_state=seed,
                    )
                ),
            ),
        )
    )
    if tuple(registry) != MODEL_ORDER:
        raise RuntimeError("Model registry order differs from the documented baseline suite.")
    return registry


def publication_sources(names: list[str]) -> np.ndarray:
    sources = []
    for name in names:
        text = str(name).strip()
        match = re.match(r"^([A-Za-z]+[0-9]{4})", text)
        sources.append(match.group(1) if match else text.split("-", 1)[0])
    return np.asarray(sources, dtype=object)


def regression_strata(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    strata, edges = pd.qcut(
        np.asarray(targets, dtype=float),
        q=NUM_STRATIFICATION_BINS,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    strata = np.asarray(strata, dtype=int)
    _, counts = np.unique(strata, return_counts=True)
    if len(counts) < 2 or int(counts.min()) < N_SPLITS:
        raise ValueError(f"Cannot construct {N_SPLITS} target-stratified folds: {counts.tolist()}")
    return strata, np.asarray(edges, dtype=float)


def source_target_strata(targets: np.ndarray, names: list[str]):
    targets = np.asarray(targets, dtype=float)
    sources = publication_sources(names)
    source_counts = pd.Series(sources).value_counts()
    buckets = np.asarray(
        [source if source_counts[source] >= 2 * N_SPLITS else "Other" for source in sources],
        dtype=object,
    )
    labels = np.empty(len(targets), dtype=object)
    local_bins = {}
    for bucket in sorted(set(buckets)):
        positions = np.flatnonzero(buckets == bucket)
        requested_bins = min(NUM_STRATIFICATION_BINS, len(positions) // N_SPLITS)
        while requested_bins >= 1:
            codes, edges = pd.qcut(
                targets[positions],
                q=requested_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            codes = np.asarray(codes, dtype=int)
            _, counts = np.unique(codes, return_counts=True)
            if len(counts) and int(counts.min()) >= N_SPLITS:
                break
            requested_bins -= 1
        if requested_bins < 1:
            raise ValueError(f"Cannot construct source-target strata for {bucket!r}.")
        for position, code in zip(positions, codes):
            labels[position] = f"{bucket}::q{int(code)}"
        local_bins[bucket] = {
            "rows": int(len(positions)),
            "bins": int(len(np.unique(codes))),
            "edges": [float(value) for value in edges],
        }
    _, global_edges = regression_strata(targets)
    return labels, global_edges, {
        "strategy": "publication_source_and_within_source_target_quantiles",
        "raw_source_counts": source_counts.sort_index().astype(int).to_dict(),
        "source_bucket_counts": pd.Series(buckets).value_counts().sort_index().astype(int).to_dict(),
        "local_target_bins": local_bins,
        "sources": sources.tolist(),
        "source_buckets": buckets.tolist(),
        "strata": labels.tolist(),
    }


def reaction_group_ids(frame: pd.DataFrame) -> np.ndarray:
    missing = [column for column in REACTION_GROUP_COLUMNS if column not in frame]
    if missing:
        raise KeyError(f"Co table is missing reaction-group columns: {missing}")
    groups = []
    for _, row in frame.iterrows():
        sample_name = str(row[NAME_COLUMN])
        catalyst = str(row["Cat"]).strip() if not pd.isna(row["Cat"]) else "Co"
        components = [catalyst or "Co"]
        for column in REACTION_GROUP_COLUMNS[1:]:
            components.append(
                _canonical_smiles(row[column], role=column, sample_name=sample_name)
            )
        identity = "\x1f".join(components).encode("utf-8")
        groups.append(hashlib.sha256(identity).hexdigest()[:16])
    return np.asarray(groups, dtype=object)


def make_co_splits(frame: pd.DataFrame, targets: np.ndarray, seed: int = 10):
    names = frame[NAME_COLUMN].astype(str).tolist()
    strata, edges, audit = source_target_strata(targets, names)
    groups = reaction_group_ids(frame)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(frame)), strata, groups=groups))
    folds = np.zeros(len(frame), dtype=int)
    for fold, (_, validation_indices) in enumerate(splits, start=1):
        folds[validation_indices] = fold
    group_fold_counts = pd.DataFrame({"group": groups, "fold": folds}).groupby("group")["fold"].nunique()
    cross_fold_groups = group_fold_counts[group_fold_counts > 1].index.tolist()
    if cross_fold_groups:
        raise RuntimeError(f"Reaction groups cross Co folds: {cross_fold_groups[:5]}")
    audit.update(
        {
            "n_splits": N_SPLITS,
            "cv_seed": int(seed),
            "reaction_group_columns": list(REACTION_GROUP_COLUMNS),
            "reaction_groups": int(len(set(groups))),
            "groups_split_across_folds": 0,
            "target_quantile_edges": [float(value) for value in edges],
            "fold_assignments": folds.tolist(),
            "reaction_group_ids": groups.tolist(),
        }
    )
    return splits, folds, groups, audit


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if targets.shape != predictions.shape or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite and have the same shape as targets.")
    return {
        "r2": float(r2_score(targets, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported table: {path}")
    required = [NAME_COLUMN, TARGET_COLUMN, *ROLE_COLUMNS.values()]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"{path.name} is missing required columns: {missing}")
    targets = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError(f"{path.name} contains non-finite {TARGET_COLUMN} values.")
    if frame[NAME_COLUMN].isna().any():
        raise ValueError(f"{path.name} contains missing sample names.")
    return frame.reset_index(drop=True)


def _cache_key(
    path: Path,
    fingerprint_bits: int,
    environment_fingerprint_bits: int,
    include_3d_descriptors: bool,
    descriptor_num_conformers: int,
    descriptor_conformer_seed: int,
    optimize_conformers: bool,
) -> dict[str, object]:
    return {
        "feature_version": FEATURE_VERSION,
        "data_path": str(path.resolve()),
        "data_sha256": _sha256(path),
        "reaction_fingerprint_bits": int(fingerprint_bits),
        "environment_fingerprint_bits": int(environment_fingerprint_bits),
        "include_3d_descriptors": bool(include_3d_descriptors),
        "descriptor_num_conformers": int(descriptor_num_conformers),
        "descriptor_conformer_seed": int(descriptor_conformer_seed),
        "optimize_conformers": bool(optimize_conformers),
        "role_columns": dict(ROLE_COLUMNS),
        "descriptor_names": list(
            molecular_descriptor_names(include_3d=bool(include_3d_descriptors))
        ),
    }


def load_or_build_features(
    frame: pd.DataFrame,
    path: Path,
    cache_dir: Path,
    label: str,
    fingerprint_bits: int,
    environment_fingerprint_bits: int,
    include_3d_descriptors: bool,
    descriptor_num_conformers: int,
    descriptor_conformer_seed: int,
    optimize_conformers: bool,
    use_cache: bool,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"{label}_features.npz"
    metadata_path = cache_dir / f"{label}_features.json"
    expected = _cache_key(
        path,
        fingerprint_bits,
        environment_fingerprint_bits,
        include_3d_descriptors,
        descriptor_num_conformers,
        descriptor_conformer_seed,
        optimize_conformers,
    )
    if use_cache and matrix_path.is_file() and metadata_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved == expected:
            with np.load(matrix_path, allow_pickle=False) as archive:
                matrix = np.asarray(archive["features"], dtype=np.float32)
            names = feature_names(
                fingerprint_bits,
                environment_fingerprint_bits,
                include_3d_descriptors,
            )
            if matrix.shape == (len(frame), len(names)) and np.isfinite(matrix).all():
                print(f"Loaded cached {label} features: {matrix.shape}")
                return matrix, names
    matrix, names = featurize_frame(
        frame,
        fingerprint_bits=fingerprint_bits,
        environment_fingerprint_bits=environment_fingerprint_bits,
        include_3d_descriptors=include_3d_descriptors,
        descriptor_num_conformers=descriptor_num_conformers,
        descriptor_conformer_seed=descriptor_conformer_seed,
        optimize_conformers=optimize_conformers,
    )
    if use_cache:
        np.savez_compressed(matrix_path, features=matrix)
        metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {label} RDKit features: {matrix.shape}")
    return matrix, names


def _predict_vector(model: BaseEstimator, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(features), dtype=float).reshape(-1)


@contextmanager
def bounded_native_threads(limit: int):

    limiter = None
    entered = False
    try:
        limiter = threadpool_limits(limits=int(limit))
        limiter.__enter__()
        entered = True
    except (OSError, PermissionError):


        limiter = None
    try:
        yield
    finally:
        if entered and limiter is not None:
            try:
                limiter.__exit__(None, None, None)
            except (OSError, PermissionError):
                pass


def evaluate_ir(
    development: pd.DataFrame,
    test: pd.DataFrame,
    development_features: np.ndarray,
    test_features: np.ndarray,
    registry,
    native_threads: int,
):
    train_targets = development[TARGET_COLUMN].to_numpy(dtype=float)
    test_targets = test[TARGET_COLUMN].to_numpy(dtype=float)
    summary, predictions, failures = [], [], []
    for model_name, model in registry.items():
        print(f"[Ir] {model_name}")
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(), bounded_native_threads(native_threads):
                warnings.simplefilter("ignore")
                model.fit(development_features, train_targets)
            predicted = _predict_vector(model, test_features)
            metrics = regression_metrics(test_targets, predicted)
            elapsed = time.perf_counter() - started
            summary.append(
                {
                    "dataset": "Ir",
                    "evaluation": "fixed_independent_10pct_holdout",
                    "model": model_name,
                    "train_rows": len(development),
                    "evaluation_rows": len(test),
                    **metrics,
                    "fit_predict_seconds": elapsed,
                }
            )
            for row_index, (name, target, prediction) in enumerate(
                zip(test[NAME_COLUMN].astype(str), test_targets, predicted)
            ):
                predictions.append(
                    {
                        "dataset": "Ir",
                        "evaluation": "fixed_independent_10pct_holdout",
                        "model": model_name,
                        "row_index": row_index,
                        "fold": "test",
                        "sample_name": name,
                        "true_delta_g_kcal_mol": target,
                        "predicted_delta_g_kcal_mol": prediction,
                        "residual_predicted_minus_true": prediction - target,
                        "absolute_error": abs(prediction - target),
                    }
                )
            print(
                f"  R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} ({elapsed:.1f}s)"
            )
        except Exception as error:
            failures.append({"dataset": "Ir", "model": model_name, "error": repr(error)})
            print(f"  FAILED: {error!r}")
    return summary, predictions, failures


def evaluate_co(
    frame: pd.DataFrame,
    features: np.ndarray,
    registry,
    splits,
    folds,
    groups,
    native_threads: int,
):
    targets = frame[TARGET_COLUMN].to_numpy(dtype=float)
    sources = publication_sources(frame[NAME_COLUMN].astype(str).tolist())
    summary, fold_metrics, predictions, failures = [], [], [], []
    for model_name, template in registry.items():
        print(f"[Co] {model_name}")
        started = time.perf_counter()
        oof = np.full(len(frame), np.nan, dtype=float)
        model_fold_rows = []
        try:
            from sklearn.base import clone

            for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
                model = clone(template)
                with warnings.catch_warnings(), bounded_native_threads(native_threads):
                    warnings.simplefilter("ignore")
                    model.fit(features[train_indices], targets[train_indices])
                fold_predictions = _predict_vector(model, features[validation_indices])
                oof[validation_indices] = fold_predictions
                values = regression_metrics(targets[validation_indices], fold_predictions)
                model_fold_rows.append(
                    {
                        "dataset": "Co",
                        "evaluation": "reaction_grouped_5fold_oof",
                        "model": model_name,
                        "fold": fold,
                        "train_rows": len(train_indices),
                        "validation_rows": len(validation_indices),
                        **values,
                    }
                )
            metrics = regression_metrics(targets, oof)
            elapsed = time.perf_counter() - started
            summary.append(
                {
                    "dataset": "Co",
                    "evaluation": "reaction_grouped_5fold_oof",
                    "model": model_name,
                    "train_rows": "fold-dependent",
                    "evaluation_rows": len(frame),
                    **metrics,
                    "fold_r2_mean": float(np.mean([row["r2"] for row in model_fold_rows])),
                    "fold_r2_std": float(np.std([row["r2"] for row in model_fold_rows], ddof=1)),
                    "fit_predict_seconds": elapsed,
                }
            )
            fold_metrics.extend(model_fold_rows)
            for row_index, (name, source, fold, group, target, prediction) in enumerate(
                zip(frame[NAME_COLUMN].astype(str), sources, folds, groups, targets, oof)
            ):
                predictions.append(
                    {
                        "dataset": "Co",
                        "evaluation": "reaction_grouped_5fold_oof",
                        "model": model_name,
                        "row_index": row_index,
                        "fold": int(fold),
                        "sample_name": name,
                        "publication_source": source,
                        "reaction_group_id": group,
                        "true_delta_g_kcal_mol": target,
                        "predicted_delta_g_kcal_mol": prediction,
                        "residual_predicted_minus_true": prediction - target,
                        "absolute_error": abs(prediction - target),
                    }
                )
            print(
                f"  OOF R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} ({elapsed:.1f}s)"
            )
        except Exception as error:
            failures.append({"dataset": "Co", "model": model_name, "error": repr(error)})
            print(f"  FAILED: {error!r}")
    return summary, fold_metrics, predictions, failures


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare 16 RDKit-only learning algorithms plus a mean reference; "
            "all methods directly predict Delta G."
        )
    )
    split_dir = PROJECT_DIR / "data" / "ir_holdout"
    parser.add_argument(
        "--ir-development", default=str(split_dir / "Database Ir Development.csv")
    )
    parser.add_argument("--ir-test", default=str(split_dir / "Database Ir Test.csv"))
    parser.add_argument("--co-data", default=str(PROJECT_DIR / "data" / "Database Co.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_DIR
            / "outputs"
            / "rdkit_ml_baselines"
            / "co_old_seed_10_compact"
        ),
    )
    parser.add_argument("--dataset", choices=("both", "ir", "co"), default="both")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated names or 'all'. Use --list-models to inspect names.",
    )
    parser.add_argument(
        "--fingerprint-bits",
        type=int,
        default=REACTION_FINGERPRINT_BITS,
        help="Compact Morgan width for ligand/reactant/product branches; default: 256.",
    )
    parser.add_argument(
        "--environment-fingerprint-bits",
        type=int,
        default=ENVIRONMENT_FINGERPRINT_BITS,
        help="Compact Morgan width for the solvent branch; default: 128.",
    )
    parser.add_argument(
        "--no-3d-descriptors",
        action="store_true",
        help="Disable the model-matched ETKDG descriptor block (diagnostics only).",
    )
    parser.add_argument(
        "--descriptor-num-conformers",
        type=int,
        default=DESCRIPTOR_NUM_CONFORMERS,
    )
    parser.add_argument(
        "--descriptor-conformer-seed",
        type=int,
        default=DESCRIPTOR_CONFORMER_SEED,
    )
    parser.add_argument(
        "--no-optimize-conformers",
        action="store_true",
        help="Keep ETKDG coordinates without the main model's MMFF/UFF relaxation.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--native-threads",
        type=int,
        default=1,
        help="OpenMP/BLAS thread limit; one is the reproducible Windows-safe default.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer ensemble estimators for code/runtime checks; not the final protocol.",
    )
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def _selected_registry(args):
    registry = build_model_registry(seed=args.seed, n_jobs=args.n_jobs, quick=args.quick)
    if args.models.strip().lower() == "all":
        return registry
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in requested if name not in registry]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {list(registry)}")
    return OrderedDict((name, registry[name]) for name in requested)


def main() -> None:
    args = parse_args()
    if args.list_models:
        print("\n".join(MODEL_ORDER))
        return
    if args.fingerprint_bits < 32:
        raise ValueError("--fingerprint-bits must be at least 32.")
    if args.environment_fingerprint_bits < 32:
        raise ValueError("--environment-fingerprint-bits must be at least 32.")
    if args.descriptor_num_conformers < 1:
        raise ValueError("--descriptor-num-conformers must be at least 1.")
    if args.native_threads < 1:
        raise ValueError("--native-threads must be at least 1.")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.native_threads))
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, str(args.native_threads))
    registry = _selected_registry(args)
    output_dir = Path(args.output_dir).resolve()
    if args.quick:
        output_dir = output_dir / "quick"
    cache_dir = output_dir / "feature_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary, all_predictions, all_fold_metrics, all_failures = [], [], [], []
    input_paths = {}
    include_3d_descriptors = not args.no_3d_descriptors
    optimize_conformers = not args.no_optimize_conformers
    schema_names = feature_names(
        args.fingerprint_bits,
        args.environment_fingerprint_bits,
        include_3d_descriptors,
    )
    feature_arguments = {
        "fingerprint_bits": args.fingerprint_bits,
        "environment_fingerprint_bits": args.environment_fingerprint_bits,
        "include_3d_descriptors": include_3d_descriptors,
        "descriptor_num_conformers": args.descriptor_num_conformers,
        "descriptor_conformer_seed": args.descriptor_conformer_seed,
        "optimize_conformers": optimize_conformers,
        "use_cache": not args.no_feature_cache,
    }

    if args.dataset in ("both", "ir"):
        development_path = Path(args.ir_development).resolve()
        test_path = Path(args.ir_test).resolve()
        development = _read_table(development_path)
        test = _read_table(test_path)
        overlap = set(development[NAME_COLUMN].astype(str)) & set(test[NAME_COLUMN].astype(str))
        if overlap:
            raise ValueError(f"Ir development/test Name overlap: {sorted(overlap)[:5]}")
        x_development, names = load_or_build_features(
            development,
            development_path,
            cache_dir,
            "ir_development",
            **feature_arguments,
        )
        x_test, _ = load_or_build_features(
            test,
            test_path,
            cache_dir,
            "ir_test",
            **feature_arguments,
        )
        if names != schema_names:
            raise RuntimeError("Ir feature schema changed within one run.")
        summary, predictions, failures = evaluate_ir(
            development,
            test,
            x_development,
            x_test,
            registry,
            args.native_threads,
        )
        all_summary.extend(summary)
        all_predictions.extend(predictions)
        all_failures.extend(failures)
        input_paths.update(
            {
                "ir_development": {
                    "path": str(development_path),
                    "sha256": _sha256(development_path),
                    "rows": len(development),
                },
                "ir_test": {
                    "path": str(test_path),
                    "sha256": _sha256(test_path),
                    "rows": len(test),
                },
            }
        )

    co_split_audit = None
    if args.dataset in ("both", "co"):
        co_path = Path(args.co_data).resolve()
        co = _read_table(co_path)
        x_co, names = load_or_build_features(
            co,
            co_path,
            cache_dir,
            "co",
            **feature_arguments,
        )
        if names != schema_names:
            raise RuntimeError("Co and Ir feature schemas differ.")
        targets = co[TARGET_COLUMN].to_numpy(dtype=float)
        splits, folds, groups, co_split_audit = make_co_splits(co, targets, args.seed)
        summary, fold_metrics, predictions, failures = evaluate_co(
            co,
            x_co,
            registry,
            splits,
            folds,
            groups,
            args.native_threads,
        )
        all_summary.extend(summary)
        all_fold_metrics.extend(fold_metrics)
        all_predictions.extend(predictions)
        all_failures.extend(failures)
        input_paths["co"] = {
            "path": str(co_path),
            "sha256": _sha256(co_path),
            "rows": len(co),
        }

    summary_frame = pd.DataFrame(all_summary)
    predictions_frame = pd.DataFrame(all_predictions)
    if not predictions_frame.empty:
        predictions_frame = export_classification(
            predictions_frame, output_dir, group_columns=("model", "evaluation"),
            role="RDKit direct-G baselines: Ir test / Co grouped OOF",
            plots=not args.skip_plots,
        )
    fold_frame = pd.DataFrame(all_fold_metrics)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            ["dataset", "r2"], ascending=[True, False]
        ).reset_index(drop=True)
    summary_frame.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(
        output_dir / "all_predictions_long.csv", index=False, encoding="utf-8-sig"
    )
    fold_frame.to_csv(output_dir / "co_fold_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "failures.json").write_text(
        json.dumps(all_failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if co_split_audit is not None:
        (output_dir / "co_cv_split_audit.json").write_text(
            json.dumps(co_split_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "row_index": np.arange(len(co_split_audit["fold_assignments"])),
                "fold": co_split_audit["fold_assignments"],
                "reaction_group_id": co_split_audit["reaction_group_ids"],
                "source": co_split_audit["sources"],
                "source_bucket": co_split_audit["source_buckets"],
                "stratum": co_split_audit["strata"],
            }
        ).to_csv(output_dir / "co_cv_split_audit.csv", index=False, encoding="utf-8-sig")

    feature_schema = {
        "version": FEATURE_VERSION,
        "rdkit_only": True,
        "matches_main_model_global_rdkit_blocks": False,
        "matches_main_model_global_rdkit_block_layout": True,
        "fingerprint_width_matches_main_model": False,
        "predicts": "Delta G directly",
        "target_column": TARGET_COLUMN,
        "temperature_in_model_inputs": False,
        "xtb_in_model_inputs": False,
        "role_columns": dict(ROLE_COLUMNS),
        "reaction_branch_morgan": {
            "radius": 2,
            "bits_per_fingerprint": int(args.fingerprint_bits),
            "include_chirality": False,
            "molecules": ["ligand", "combined reactant or product"],
        },
        "environment_branch_morgan": {
            "radius": 2,
            "bits_per_fingerprint": int(args.environment_fingerprint_bits),
            "include_chirality": False,
            "molecules": ["solvent"],
        },
        "rdkit_descriptors": {
            "include_3d": include_3d_descriptors,
            "num_conformers": int(args.descriptor_num_conformers),
            "conformer_seed": int(args.descriptor_conformer_seed),
            "optimize_conformers": optimize_conformers,
            "names_per_molecule": list(
                molecular_descriptor_names(include_3d=include_3d_descriptors)
            ),
        },
        "flattened_branch_order": [
            "environment: solvent fingerprint + solvent descriptors",
            (
                "reactant: ligand fingerprint + combined substrate/boron/ligand "
                "fingerprint + ligand descriptors + combined-reactant descriptors"
            ),
            (
                "product: ligand fingerprint + product fingerprint + ligand "
                "descriptors + product descriptors"
            ),
        ],
        "ligand_block_is_repeated": (
            "yes; it appears in both reactant and product global inputs, matching the GNN"
        ),
        "feature_dimension": len(schema_names),
        "fold_local_feature_reduction": {
            "constant_feature_filter": "VarianceThreshold fitted on training rows only",
            "method": "SelectKBest(f_regression) fitted on training rows only",
            "default_max_selected_features": SELECTED_FEATURES,
            "bayesian_ridge_pls_knn_max_selected_features": 128,
        },
        "feature_names": schema_names,
        "explicitly_excluded_table_columns": list(FORBIDDEN_MODEL_COLUMNS),
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    protocol = {
        "task": "RDKit-only classical ML direct Delta-G regression",
        "learning_algorithms": 16,
        "mean_only_reference_models": 1,
        "quick_mode": bool(args.quick),
        "quick_mode_is_final_result": False if args.quick else None,
        "models_requested": list(registry),
        "models_completed": sorted(set(summary_frame.get("model", []))),
        "parallelism": {
            "sklearn_n_jobs": int(args.n_jobs),
            "native_thread_limit": int(args.native_threads),
        },
        "compact_tree_capacity": {
            "bagging_estimators": int(
                QUICK_ENSEMBLE_TREES if args.quick else ENSEMBLE_TREES
            ),
            "boosting_iterations": int(
                QUICK_BOOSTING_STEPS if args.quick else BOOSTING_STEPS
            ),
            "tree_input_features_at_most": SELECTED_FEATURES,
            "random_forest_extra_trees_max_depth": 10,
            "random_forest_extra_trees_max_leaf_nodes": 64,
        },
        "input_files": input_paths,
        "feature_schema_file": "feature_schema.json",
        "ir_evaluation": (
            "frozen 2251-row development / 251-row independent test; test not used for fitting"
            if args.dataset in ("both", "ir")
            else None
        ),
        "co_evaluation": (
            "five-fold StratifiedGroupKFold; reaction groups exclude solvent, T and Delta G"
            if args.dataset in ("both", "co")
            else None
        ),
        "hyperparameter_policy": (
            "fixed a priori in source code; no test/validation tuning inside this run"
        ),
        "multiple_model_warning": (
            "Selecting the best of many models from these reported scores introduces selection bias; "
            "confirm a chosen model on a new external test set."
        ),
        "failures": all_failures,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nCompleted {len(summary_frame)} dataset/model evaluations.")
    print(f"Summary: {output_dir / 'summary_metrics.csv'}")
    if all_failures:
        print(f"Failures: {len(all_failures)}; inspect {output_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
