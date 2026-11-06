# Contributing

Read [README.md](README.md) for the scientific protocol and installation steps.
Before submitting a change, describe the problem, expected behavior and a minimal
reproduction in an issue or pull request. Avoid including credentials or private data.

Run commands from the repository root:

```sh
python -m pip install -e ".[dev]" --no-deps
python -m compileall -q src scripts analysis baselines
python -m pytest src/test_suite.py src/shap/test_grouped_shapley.py -q
```

Install compatible scientific dependencies and pytest separately when using
`--no-deps`. Full model tests require PyTorch, DGL, RDKit and the supplied assets.
GitHub CI runs syntax checks and classification tests.

`src/test_suite.py` contains the full test suite. The environment variable
`MCGNN_TEST_MODE=lightweight` selects classification tests without importing
graph-model dependencies; leave it unset to run the full suite.

Keep changes focused. Explain validation and any effect on predictions, split
membership or reported metrics. Preserve frozen data splits and published model
weights unless a change explicitly documents a new scientific version. Use a new
output directory for experiments rather than overwriting publication results.

Model weights use Git LFS. Do not commit local environments, caches, secrets or
temporary runs. See [release preparation](docs/RELEASE.md) for asset policy and
asset redistribution considerations. Project code uses the custom non-commercial research license in LICENSE; preserve its attribution and citation requirements.
