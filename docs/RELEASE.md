# Repository assets and distribution

Project code uses the [MCGNN Non-Commercial Research License](../LICENSE).
Use is limited to non-commercial scientific research and academic education.
Research outputs must acknowledge the authors and cite the accompanying article.
Until the article is published, cite the repository URL and version used.

## Assets

| Path | Contents |
| --- | --- |
| src, scripts, analysis, baselines | Source code and entry points |
| data, config | Datasets, feature inputs, partitions and configuration |
| models | Git LFS model checkpoints |
| outputs | Numerical results and publication figures |
| environment_versions.json | Package version information |

## Download

```sh
git lfs install
git clone <repository-url>
cd <repository-directory>
git lfs pull
```

## Upload

```sh
git init
git lfs install
git add .
git lfs ls-files
git diff --cached --stat
```

## Checks

```sh
python -m compileall -q src scripts analysis baselines
python -m pytest src/test_suite.py src/shap/test_grouped_shapley.py -q
```
