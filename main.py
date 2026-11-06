import argparse
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
MODULES = {
    "co-joint": ("src/Co-transfer/Cotrans/run_fixed_transfer.py", "Co joint-loss transfer"),
    "co-baselines": ("src/Co-transfer/co_transfer_holdout/run_co_holdout.py", "Co baselines (resume enabled)"),
    "ir-graph-space": ("analysis/graph_space/analyze_graph_space.py", "Ir graph-space analysis"),
    "ir-feature-importance": ("analysis/feature_importance/run_ir_importance.py", "Ir feature permutation importance"),
    "ir-atom-bond-importance": ("analysis/atom_bond_importance/run_ir_atom_bond_importance.py", "Ir atom/bond attribution"),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use python main.py MODULE --help for module-specific options.",
    )
    parser.add_argument("module", choices=MODULES, help="Module to run")


    selected = parser.parse_args(arguments[:1]).module
    forwarded = arguments[1:]
    if selected == "co-baselines" and "--resume" not in forwarded:
        forwarded.append("--resume")
    entry = ROOT / MODULES[selected][0]
    namespace = runpy.run_path(str(entry), run_name="publication_module")
    return namespace["main"](forwarded)


if __name__ == "__main__":
    main()
