from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT, ROOT / "src/Co-transfer", ROOT / "src/Co-transfer/Cotrans"):
    sys.path.insert(0, str(folder))

import run_fixed_transfer as engine
import fixed_protocol
from co_transfer_holdout.protocol import dump, read_json, sha256, verify_split
from experiment_protocol import verified_copy
from joint_protocol import assert_checkpoint_protocol as check_joint_checkpoint

ROLE = "Graph ablation; corresponding Ir-masked checkpoint; exact seed-126 125/14 split, joint-loss five-fold OOF and holdout"
CO_DIRECTORY = "Co_seed_126_test_14"


def output_path(args):
    mask = tuple(args.ablate_branches)
    if len(mask) not in (1, 2) or len(set(mask)) != len(mask):
        raise ValueError("Select one or two distinct graph branches")
    args.ablate_branches = sorted(mask)
    variant = "_".join(b for b in ("environment", "reactant", "product") if b in mask)
    suite = "graph_ablations" if len(mask) == 1 else "graph_pair_ablations"
    tag = ("smoke_" if args.smoke else "") + CO_DIRECTORY
    output = (args.output_dir or ROOT / "outputs/interpretability" / suite / variant / tag).resolve()
    if (ROOT / "outputs").resolve() not in output.parents or output.name != tag:
        raise ValueError(f"Use a new ablation directory below outputs ending in {tag}; old Co/ results are read-only")
    for relative in ("co_transfer", "baselines/co_transfer_holdout", "07_runtime_cache/co_graph_cache"):
        protected = (ROOT / "outputs" / relative).resolve()
        if output == protected or protected in output.parents:
            raise ValueError("Do not write ablations into main-model/baseline/cache directories")
    return output


def configure(parser):
    parser.description = __doc__
    parser.add_argument("--ablate-branches", nargs="+", required=True,
                        choices=("environment", "reactant", "product"))


def parse_args(argv=None):
    return engine.parse_args(argv, configure=configure, output_resolver=output_path)


def validate_ir_source(path, disabled):
    import torch
    from src.ir_protocol import ir_run_paths
    from src.data_splits import checkpoint_split_config
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    mask = set(metadata.get("model_config", {}).get("ablate_branches", []))
    declared = set(metadata.get("graph_ablation", {}).get("disabled_branches", []))
    if metadata.get("default_catalyst") != "Ir" or metadata.get("head") != "enthalpy_entropy_physics":
        raise ValueError("Start from the corresponding Ir ablation checkpoint, never Co weights")
    if mask != set(disabled) or declared != mask:
        raise ValueError(f"Ir checkpoint graph mask {sorted(mask)} differs from requested {sorted(disabled)}")
    if checkpoint["model_state_dict"]["fc2.weight"].shape[0] != 2:
        raise ValueError("Expected Delta H/Delta S output head")
    manifest = read_json(ir_run_paths(ROOT)["manifest"])
    if metadata.get("holdout_split") != checkpoint_split_config(manifest):
        raise ValueError("Ir ablation must use the same frozen development/test split as the main model")
    return sha256(path)


def prepare_inputs(args):
    source_hash = validate_ir_source(args.checkpoint, args.ablate_branches)
    fixed_protocol.validate_source_data(args.data)
    historical, development, test, folds = fixed_protocol.reference_tables()
    manifest = dict(historical)
    manifest.update(source=str(args.data.resolve()), source_sha256=sha256(args.data),
                    version="co_graph_ablation_fixed_seed_126_v1", mode="holdout",
                    reference_manifest_sha256=sha256(fixed_protocol.REFERENCE / "manifest.json"),
                    historical_split_was_search_selected=True,
                    sampling="Exact main-model seed-126 split and CV-seed-10 folds; no redraw")
    frozen = dict(source_checkpoint_sha256=source_hash, source_data_sha256=sha256(args.data),
                  graph_cache_key=args.cache_key, ablate_branches=args.ablate_branches)
    destination = args.output_dir / "split/manifest.json"
    if destination.exists() and read_json(destination) != manifest:
        raise ValueError("Ablation split changed; choose a new output directory")
    for name in fixed_protocol.SPLIT_FILES:
        verified_copy(fixed_protocol.REFERENCE / name, args.output_dir / "split" / name)
    if not destination.exists():
        dump(destination, manifest)
    verify_split(args.output_dir)
    frozen["split_manifest_sha256"] = sha256(destination)
    inputs = args.output_dir / "frozen_inputs.json"
    if inputs.exists() and read_json(inputs) != frozen:
        raise ValueError("Ablation source, mask or graph cache changed; use a new output directory")
    verified_copy(args.checkpoint, args.output_dir / "inputs/ir_source_checkpoint.pth")
    if not inputs.exists():
        dump(inputs, frozen)
    return development, test, folds


def assert_checkpoint_protocol(metadata, output, fold):
    check_joint_checkpoint(metadata, output, fold)
    disabled = set(read_json(output / "frozen_inputs.json")["ablate_branches"])
    if set(metadata.get("model_config", {}).get("ablate_branches", [])) != disabled:
        raise ValueError("Co checkpoint graph mask differs from frozen ablation")
    if set(metadata.get("graph_ablation", {}).get("disabled_branches", [])) != disabled:
        raise ValueError("Co checkpoint ablation declaration differs")


def code_paths():
    return [Path(__file__).resolve()]


def main(argv=None):
    engine.main(argv, experiment=sys.modules[__name__])


if __name__ == "__main__":
    main()
