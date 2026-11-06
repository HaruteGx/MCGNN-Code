import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.data_splits import ensure_random_ir_split
from src.ir_protocol import TEST_SPLIT_SEED, TEST_FRACTION, ir_run_paths


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    paths = ir_run_paths(PROJECT_DIR)
    development_path, test_path = paths["development"], paths["test"]
    development, test, audit = ensure_random_ir_split(
        paths["full"], development_path, test_path, paths["manifest"],
        seed=TEST_SPLIT_SEED, test_fraction=TEST_FRACTION)
    print(f"Development table: {development_path}")
    print(f"Test table: {test_path}")


if __name__ == "__main__":
    main()
