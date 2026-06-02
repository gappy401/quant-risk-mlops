# DVC — Reproducible Pipeline

> Rundown page for the **DVC** (Data Version Control) layer of `quant-risk-mlops`, in the same shape as the other technology pages.

## What DVC is

DVC is version control for data and pipelines, designed to sit next to git. Git is great for code but bad at large or binary files, and it has no notion of "rerun the steps that produced this." DVC fills both gaps. It tracks large data and model files by hash (storing them outside git), and it defines your processing steps as a pipeline that can be rerun reproducibly. The slogan is "git for data and pipelines."

## What it does

1. **Defines a pipeline** as a set of stages, each with its inputs, parameters, and outputs.
2. **Reruns only what changed.** `dvc repro` checks every stage and executes only the ones whose inputs moved, like `make` for data science.
3. **Tracks data and model outputs by hash**, keeping them out of git while still versioning them.
4. **Records exact state** in a lock file, so you know precisely what produced the current outputs.
5. **Compares metrics** across runs.

## Core concepts

1. **Stage**: one step in the pipeline (a command plus its inputs and outputs).
2. **deps**: a stage's inputs, both data files and source code. A change here invalidates the stage and everything downstream.
3. **params**: tracked knobs read from `params.yaml`, interpolated into the command.
4. **outs**: files the stage produces, which DVC takes ownership of and keeps out of git.
5. **metrics**: tracked numbers (kept in the workspace) that you can compare across runs.
6. **dvc.yaml**: the pipeline definition (the stages and their wiring).
7. **dvc.lock**: the record of the exact inputs and outputs (by hash) of the last run.
8. **The git and DVC split**: code, `dvc.yaml`, `dvc.lock`, and `params.yaml` go in git; the large data and model outputs are tracked by DVC.

## The commands (and how to read them)

The pattern is `dvc VERB [ARGS]`.

1. `dvc init` sets up DVC in the repo (creates a `.dvc/` folder). Run once.
2. `dvc repro` runs the pipeline, executing only stages whose deps or params changed. Run it again and unchanged stages are skipped.
3. `dvc dag` prints the stage dependency graph.
4. `dvc metrics show` displays the tracked metrics, and `dvc metrics diff` compares them across versions.
5. `dvc status` shows what is out of date.
6. `dvc push` and `dvc pull` move the tracked data to and from remote storage (for example S3), the data equivalent of git push and pull.

## How it fits this project, and where it comes alive

`dvc.yaml` wires three stages into one graph: generate, then train, then monitor. `params.yaml` holds the knobs that drive them, such as the synthetic row count and the drift level, and those values are interpolated into the stage commands so changing a knob is tracked. Each stage declares its deps (including the source code under `src/quant_risk`), so editing the code or a parameter invalidates exactly the right stages and everything downstream. `dvc.lock` records what produced the current outputs, which is the pipeline level twin of the data hash logged in MLflow: between the two, any model traces back to the exact data and code that made it.

Where it comes alive is the second `dvc repro`. The first run executes all three stages; the second prints "Data and pipelines are up to date" and skips everything, because nothing changed. Then change `n_synthetic` in `params.yaml` and rerun: generate reruns, which changes its output, which forces train, which forces monitor, a clean cascade. That is reproducibility enforced mechanically. You never recompute what did not change, and you can never forget to recompute what did.

## Documentation links

1. DVC documentation home: https://dvc.org/doc
2. Get started, data pipelines: https://dvc.org/doc/start/data-pipelines/data-pipelines
3. dvc.yaml reference: https://dvc.org/doc/user-guide/project-structure/pipelines-files
4. dvc repro reference: https://dvc.org/doc/command-reference/repro
