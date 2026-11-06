# MCGNN for metal-catalyzed C-H borylation

Publication code and numerical data for Ir pretraining and Co transfer learning.
Run all commands from this directory. This folder is self-contained and can be copied to a separate repository.

## License

This project's code is available under the custom
[MCGNN Non-Commercial Research License](LICENSE). Use is limited to
non-commercial scientific research and academic education. Commercial use and
use for financial gain are prohibited without prior written permission.

Research outputs using this code must acknowledge the authors and cite the
accompanying article. The article is currently unpublished; until its citation
is available, cite this repository URL and the version or commit used.
Once the article citation is supplied, cite it in subsequent research outputs.

## Contents

- `src`: three graph encoders, catalyst context, features, thermodynamic output, training, prediction and the unified `test_suite.py` test module.
- `src/Co-transfer`: fixed Co split, two-stage joint-loss transfer and common baseline protocol.
- `baselines`: RDKit regression baselines and graph-branch ablations.
- `analysis`: graph representation, feature permutation and conditional atom/bond integrated gradients.
- `src/shap`: grouped Monte Carlo Shapley analysis, with temperature fixed and an option to fix fingerprints.
- `data`: the 2,502-record Ir and 139-record Co datasets and frozen Ir development/test split.
- `data/xtb_*_features`: molecular manifests and the six raw file types actually read by the feature parser.
- `models`: `Ir checkpoints.pth` (Ir pretrained model) and `Co transfer checkpoints.pth` (selected Co transfer model).
- `outputs`: numerical result tables and publication figures. SHAP results are under `outputs/interpretability/shap`.

Ir classical ML results are under `outputs/baselines/ML`. Co baseline results are under `outputs/baselines/co_transfer_holdout/evaluations`.

## Environment

Use Linux x86_64 or Windows WSL2 with an NVIDIA GPU and a driver supporting
CUDA 11.8. Install [Miniforge](https://github.com/conda-forge/miniforge/releases),
then run these commands in a Linux or WSL2 terminal from the repository root:

```sh
conda create -n mcgnn python=3.9.18 pip -y
conda activate mcgnn
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
python -c "import torch, dgl, rdkit; print(torch.__version__, dgl.__version__, rdkit.__version__); print('CUDA available:', torch.cuda.is_available())"
```

[requirements.txt](requirements.txt) specifies package versions and official
PyTorch/DGL download sources. Python is installed by the `conda create` command.
The CUDA 11.8 wheels provide the runtime libraries; the host supplies the NVIDIA driver.
For Windows, install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
and run the commands inside its Linux environment.

To download the package files without installing them, run this inside the
activated Python 3.9 environment:

```sh
python -m pip download -r requirements.txt -d wheelhouse
```

Official packages: [PyTorch](https://pytorch.org/get-started/previous-versions/)
and [DGL CUDA 11.8 wheels](https://data.dgl.ai/wheels/cu118/repo.html).

## Predict

Prepare or validate the frozen Ir split without training: `python scripts/prepare_ir_test_set.py`.

```sh
python scripts/predict.py --model "models/Ir checkpoints.pth" --input "data/ir_holdout/Database Ir Test.csv" --output outputs/ir_predictions.csv --cpu
python scripts/evaluate_ir_test.py --cpu
python scripts/predict.py --model "models/Co transfer checkpoints.pth" --input config/co_reference/test.csv --xtb_root data/xtb_Co_features --default_catalyst Co --output outputs/co_predictions.csv --cpu
```

Input column names and reaction-site indices follow the supplied CSVs. Temperature is provided in Celsius and converted to Kelvin internally. Missing xTB values retain explicit availability masks. Predicting new molecules with full features requires preparing their xTB files in the same schema; the supplied feature files cover the published data.

## Reproduce training and analysis

Analysis commands export graph-space coordinates, neighbor tables,
permutation-importance summaries, atom/bond attribution tables and SHAP estimates.
Basic training curves and regression/classification evaluation figures are supported.
Use a separate output directory for each analysis run.

```sh
python scripts/train_full_model.py
python main.py co-joint --prepare-only
python main.py co-joint
python main.py co-baselines --prepare-only
python main.py co-baselines --resume
python baselines/ML/train_rdkit_baselines.py
python baselines/ablation/run_graph_ablations.py --skip-co
python main.py ir-graph-space --help
python main.py ir-feature-importance --help
python main.py ir-atom-bond-importance --help
python src/shap/run_shap.py --fingerprints fixed --output-dir outputs/interpretability/shap/reproduced_nonfingerprint
python src/shap/run_shap.py --output-dir outputs/interpretability/shap/reproduced_allfeatures
python -m pytest src/test_suite.py src/shap/test_grouped_shapley.py -q
```

## Directory interfaces

Use `main.py` to select a training or analysis module.
Run `python main.py --help` to list modules. Append `--help` after a module
to see its options; other arguments are passed to that module unchanged.
Co baselines automatically enable `--resume`.

| Module | Command |
| --- | --- |
| Co joint-loss transfer | `python main.py co-joint` |
| Co baselines | `python main.py co-baselines` |
| Ir graph-space analysis | `python main.py ir-graph-space` |
| Ir feature importance | `python main.py ir-feature-importance` |
| Ir atom/bond importance | `python main.py ir-atom-bond-importance` |

The Ir ablation output directory is `outputs/interpretability/graph_ablations/<branch>/`; training and summary entrypoints use the same location. New permutation-importance runs go under `outputs/interpretability/feature_importance/`. Explicit `--output-dir` values are resolved relative to the calling directory; use absolute paths when launching from elsewhere. Prediction accepts quoted checkpoint filenames containing spaces.
