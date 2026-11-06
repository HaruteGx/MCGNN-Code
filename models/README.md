# Published checkpoints

Model checkpoints are hosted on [Hugging Face](https://huggingface.co/LiangGroup/MCGNN_Coding/tree/main). Download the checkpoint files listed below and place them in the `models/` directory before running predictions.

- `Ir checkpoints.pth`: Ir pretrained model; selected CV fold 5; fixed 251-record test set.
- `Co transfer checkpoints.pth`: Co transfer model initialized from Ir; selected fold 1, stage 2; fixed 14-record test set.

Both models use the same three-graph architecture and latent H/S head; select the matching xTB directory and catalyst when predicting.

```sh
python scripts/predict.py --model "models/Ir checkpoints.pth" --input "data/ir_holdout/Database Ir Test.csv" --output outputs/ir_predictions.csv --cpu
python scripts/predict.py --model "models/Co transfer checkpoints.pth" --input config/co_reference/test.csv --xtb_root data/xtb_Co_features --default_catalyst Co --output outputs/co_predictions.csv --cpu
```

Run commands from the repository root.
