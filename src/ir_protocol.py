from pathlib import Path


def load_ir_test_predictions(project_dir, test):

    import numpy as np
    import pandas as pd

    path = Path(project_dir) / "outputs/ir_prediction/tables/ir_models_predictions.csv"
    frame = pd.read_csv(path, dtype={"Sample Name": str})
    frame = frame.loc[frame.model.eq("Regress") & frame.evaluation.eq("existing_test")].copy()
    frame = frame.rename(columns={"Sample Name": "Name"}).set_index("Name")
    names = test.Name.astype(str)
    if not frame.index.is_unique or set(frame.index) != set(names):
        raise ValueError("Saved Ir predictions do not match the frozen test identities")
    frame = frame.loc[names]
    if "status" in frame and not frame.status.eq("OK").all():
        raise ValueError("Saved Ir predictions contain unsuccessful rows")
    columns = ["True G (kcal/mol)", "Predicted G (kcal/mol)"]
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Saved Ir predictions contain nonfinite values")
    np.testing.assert_allclose(frame[columns[0]], test["G (kcal/mol)"], rtol=0, atol=2e-6)
    return frame


TEST_SPLIT_SEED = 10
TEST_FRACTION = 0.10


def ir_run_paths(project_dir, seed=None, test_fraction=None):
    seed = TEST_SPLIT_SEED if seed is None else seed
    test_fraction = TEST_FRACTION if test_fraction is None else test_fraction
    tag = f"seed_{int(seed)}_test_{float(test_fraction)*100:.8g}pct".replace(".", "p")
    project = Path(project_dir)
    split = project / "data" / "ir_holdout"
    run = project / "outputs" / "ir_training" / tag
    return {
        "tag": tag, "full": project / "data" / "Database Ir.csv",
        "development": split / "Database Ir Development.csv",
        "test": split / "Database Ir Test.csv", "manifest": split / "ir_holdout_split.json",
        "output": run, "best_checkpoint": run / "best_ir_xtb_metal_regression_checkpoint.pth",
        "stable_checkpoint": project / "models" / "Ir checkpoints.pth",
        "co_stable_checkpoint": project / "models" / "Co transfer checkpoints.pth",
    }
