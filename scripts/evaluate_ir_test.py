from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    args=p.parse_args()
    import torch
    from src.data_splits import load_frozen_ir_split
    from src.ir_protocol import ir_run_paths
    from src.holdout_eval import evaluate_frozen_holdout
    paths=ir_run_paths(ROOT)
    _,test,manifest=load_frozen_ir_split(paths["development"],paths["test"],paths["manifest"],paths["full"])
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    report,_=evaluate_frozen_holdout(paths["stable_checkpoint"],test,manifest,device,ROOT/"outputs/reproduced_ir_test",batch_size=args.batch_size)
    print(report)
if __name__ == "__main__":
    main()
