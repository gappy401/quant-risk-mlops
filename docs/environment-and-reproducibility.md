# Environment, Packaging, and Reproducibility

> Cross cutting page for `quant-risk-mlops`. It explains the isolated environment, why `pyproject.toml` is the project's single source of truth, how the project is reproducible end to end, and what is intentionally kept out of version control.

## The virtual environment

A virtual environment is a private, project local copy of Python and its installed packages, created with `python -m venv .venv`. You activate it (`.\.venv\Scripts\Activate.ps1` on Windows PowerShell) and from then on `python` and `pip` operate inside that sandbox.

Why it matters:

1. **Isolation.** This project's exact dependency versions live in `.venv`, separate from system Python and from other projects, so versions cannot collide.
2. **Recreatability.** The environment is disposable. You never commit it; you regenerate it from `pyproject.toml` on any machine, which is what makes "clone and run" work.
3. **Honesty.** Because the environment is clean, a missing dependency fails immediately rather than silently working off some package that happened to be on the system.

Nuances worth knowing:

1. On Windows PowerShell, activation may require allowing local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
2. The Python version itself is part of reproducibility. This project develops on Python 3.14, but PySpark does not support it yet, so the Spark phase will use a separate 3.11 or 3.12 environment, and the Docker image deliberately pins Python 3.12 so the runtime is identical everywhere regardless of the host.

## pyproject.toml, the single source of truth

`pyproject.toml` is the one standard file (PEP 621) that describes the project: its name and version, its dependencies, optional dependency groups, the build system, and tool configuration for pytest and ruff. It replaces the older mix of `setup.py`, `requirements.txt`, and scattered config.

Why we use it this way:

1. **Declarative dependencies.** The libraries the project needs are listed in one place, so the environment can be rebuilt exactly with `pip install -e ".[dev]"`.
2. **The src layout.** The package lives under `src/quant_risk`, and `pyproject.toml` tells the build to look there. This forces all imports to go through the installed package rather than loose files in the working directory, which catches "works in the repo, breaks once installed" bugs early.
3. **Editable install.** `pip install -e .` installs the package in editable mode, so source edits take effect without reinstalling, while still behaving like a real installed package.
4. **Dependency groups.** Runtime dependencies are separate from the `dev` group (pytest, ruff, and so on), so a production install does not drag in test tooling.
5. **A lean serving list.** For the Docker image, `requirements-serve.txt` lists only what the API needs at runtime (no PySpark, no DVC, no dev tools), which keeps the image small and fast.

## How reproducibility is enforced, end to end

Reproducibility here is not one trick, it is several layers that together let any result be traced back to the exact inputs and code that produced it:

1. **Declared dependencies** in `pyproject.toml` rebuild the same environment.
2. **An isolated venv** keeps that environment uncontaminated.
3. **A pinned Python version** in the container removes "works on my machine" runtime drift.
4. **A fixed random seed** in `conf/config.yaml` makes data generation, the train and test split, and model fitting deterministic.
5. **A data hash** logged with every training run records exactly which dataset produced a model.
6. **A schema version** pins the data contract the model was trained against.
7. **The DVC pipeline and `dvc.lock`** record the exact inputs and outputs of each stage, so `dvc repro` reproduces them and reruns only what changed.
8. **MLflow** logs the parameters, metrics, and the model itself for every run, giving full lineage.
9. **The container image** packages the runtime so it is identical on a laptop, in CI, and on Kubernetes.

The throughline: version the recipe, not the output. Given the code, the config, the pipeline definition, and the lock file, anyone can regenerate the data, the model, and the metrics.

## What git tracks, and what it ignores

The `.gitignore` split follows directly from "version the recipe, not the output."

Committed to git (small, text, diffable, the recipe):

1. All source code under `src/`.
2. `pyproject.toml` and `requirements-serve.txt` (how to build the environment and the image).
3. `conf/config.yaml` (settings).
4. `dvc.yaml`, `params.yaml`, and `dvc.lock` (the pipeline and its recorded state).
5. `metrics.json` (small, worth diffing across runs).
6. `docker/`, `k8s/`, and `docs/`.

Kept out of git (large, generated, or local, the output):

1. `.venv/`, because it is rebuilt from `pyproject.toml`.
2. `data/`, because it is large and managed by DVC, not git.
3. The local MLflow store (`mlflow.db`, `mlruns/`, `mlartifacts/`), because it is a running record, not source.
4. `__pycache__/` and `*.pyc`, because they are build byproducts.
5. The DVC cache under `.dvc/cache`.

The reasoning is consistent throughout: anything that can be regenerated from the committed recipe does not belong in git, because committing it bloats history and invites the version in git to drift from the version that was actually produced.

## Documentation links

1. Python virtual environments (venv): https://docs.python.org/3/library/venv.html
2. Writing a pyproject.toml: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
3. PEP 621, project metadata in pyproject.toml: https://peps.python.org/pep-0621/
4. setuptools package discovery and the src layout: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
