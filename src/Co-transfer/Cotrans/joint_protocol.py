from pathlib import Path
import numpy as np
import pandas as pd
from co_transfer_holdout.protocol import sha256


def assert_checkpoint_protocol(metadata, output, fold):
    development = pd.read_csv(output / "split/development.csv")
    audit = pd.read_csv(output / "split/development_folds.csv").set_index("Name").loc[development.Name]
    np.testing.assert_array_equal(metadata["train_indices"], np.flatnonzero(audit.fold.to_numpy() != fold))
    np.testing.assert_array_equal(metadata["validation_indices"], np.flatnonzero(audit.fold.to_numpy() == fold))
    if metadata.get("task") != "co_transfer_joint_regression_classification" or metadata.get("cv_fold") != fold:
        raise ValueError("Expected a joint-loss fold checkpoint")
    if metadata.get("split_manifest_sha256") != sha256(output / "split/manifest.json"):
        raise ValueError("Checkpoint split differs")
    if Path(metadata.get("data_path", "")).resolve() != (output / "split/development.csv").resolve():
        raise ValueError("Checkpoint development data differs")
    if metadata.get("source_checkpoint_sha256") != sha256(output / "inputs/ir_source_checkpoint.pth"):
        raise ValueError("Checkpoint Ir source differs")
