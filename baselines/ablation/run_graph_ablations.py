import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "interpretability/graph_ablations"
IR_TRAINER = PROJECT_DIR / "scripts" / "train_full_model.py"
CO_TRANSFER = Path(__file__).resolve().with_name("run_co_ablation.py")
SUMMARIZER = Path(__file__).resolve().with_name("summarize_graph_ablations.py")
BRANCHES = ("environment", "reactant", "product")


def commands_for(branch, output_root, *, skip_co, ir_epochs, stage1_epochs,
                 stage2_epochs, cpu, ablate_branches=None, skip_ir=False, prepare_co_only=False):
    disabled = tuple(ablate_branches) if ablate_branches is not None else (branch,)
    if not disabled or len(set(disabled)) != len(disabled) or set(disabled) - set(BRANCHES):
        raise ValueError(f"Invalid graph branches to ablate: {disabled}")
    ablation_args = (
        ["--ablate-branch", disabled[0]] if len(disabled) == 1
        else ["--ablate-branches", *disabled]
    )
    branch_root = output_root / branch
    ir_root = branch_root
    co_root = branch_root / "Co_seed_126_test_14"
    ir_checkpoint = ir_root / "best_ir_xtb_metal_regression_checkpoint.pth"
    ir_command = [
        sys.executable,
        str(IR_TRAINER),
        "--output-dir", str(ir_root),
        "--stable-model-path", str(ir_root / "ablation_ir_stable_checkpoint.pth"),
        *ablation_args,
        "--epochs", str(ir_epochs),
    ]
    if cpu:
        ir_command.append("--cpu")
    commands = [] if skip_ir else [(f"Ir {branch} ablation", ir_command)]
    if not skip_co:
        co_command = [
            sys.executable,
            str(CO_TRANSFER),
            "--ablate-branches", *disabled,
            "--data", str(PROJECT_DIR / "data/Database Co.csv"),
            "--checkpoint", str(ir_checkpoint),
            "--output-dir", str(co_root),
            "--split-seed", "126", "--cv-seed", "10", "--training-seed", "666",
            "--stage1-epochs", str(stage1_epochs),
            "--stage2-epochs", str(stage2_epochs),
        ]
        if prepare_co_only:
            co_command.append("--prepare-only")
        if cpu:
            co_command.append("--cpu")
        commands.append((f"Co transfer from {branch} ablation", co_command))
    return commands


def run_ablation_suite(argv=None, *, variants, output_root_default, summarizer,
                       selection_option="--branches", description):

    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(selection_option, dest="variants", nargs="+",
                        choices=tuple(variants), default=tuple(variants))
    parser.add_argument("--output-root", type=Path, default=output_root_default)
    parser.add_argument("--ir-epochs", type=int, default=100)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=150)
    parser.add_argument("--skip-co", action="store_true", help="Run Ir ablations only.")
    parser.add_argument("--skip-ir", action="store_true", help="Reuse saved Ir ablation weights; train only the original Co data.")
    parser.add_argument("--prepare-co-only", action="store_true", help="Verify Ir weights and freeze the Co 125/14 split without training or summarizing.")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of the configured accelerator.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training.")
    args = parser.parse_args(argv)
    if args.prepare_co_only:
        if args.skip_co:
            parser.error("--prepare-co-only cannot be combined with --skip-co")
        args.skip_ir = True
    if args.skip_co and args.skip_ir:
        parser.error("Cannot skip both Ir and Co")
    if min(args.ir_epochs, args.stage1_epochs, args.stage2_epochs) < 1:
        parser.error("Epoch limits must be positive integers.")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Select each ablation variant only once.")
    output_root = args.output_root.resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    for branch in args.variants:
        for label, command in commands_for(
            branch,
            output_root,
            skip_co=args.skip_co,
            ir_epochs=args.ir_epochs,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            cpu=args.cpu,
            ablate_branches=variants[branch],
            skip_ir=args.skip_ir,
            prepare_co_only=args.prepare_co_only,
        ):
            print(f"\n===== {label} =====", flush=True)
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_DIR, check=True)

    if args.prepare_co_only:
        return
    summary_command = [
        sys.executable,
        str(summarizer),
        "--root", str(output_root),
        selection_option, *args.variants,
    ]
    if args.skip_co:
        summary_command.append("--skip-co")
    print("\n===== Summarize completed ablations =====", flush=True)
    print(subprocess.list2cmdline(summary_command), flush=True)
    if not args.dry_run:
        subprocess.run(summary_command, cwd=PROJECT_DIR, check=True)


def main(argv=None):
    run_ablation_suite(
        argv,
        variants={branch: (branch,) for branch in BRANCHES},
        output_root_default=OUTPUT_ROOT,
        summarizer=SUMMARIZER,
        description=(
            "Train the three graph-branch ablations, evaluate Ir fixed holdout, "
            "transfer each ablated Ir checkpoint to Co, and summarize both tasks."
        ),
    )


if __name__ == "__main__":
    main()
