# quant-risk-mlops

An open source, portable **MLOps platform** for consumer credit **Probability of Default (PD)** scoring. A model is trained, gated, registered, served behind a real time API and a batch job, watched for drift, made reproducible with a versioned pipeline, then containerized and run on Kubernetes with autoscaling.

> The model is the payload, not the product. It is a plain logistic regression on purpose. The work that matters is the platform around it: a versioned data contract, one feature definition shared by training and serving, a model registry, drift monitoring with alerting, reproducibility, and the path from laptop to container to cluster.

A cloud agnostic build. An Azure native sibling (Databricks, ADF, ADLS, Azure ML) is planned.

## How it works, top to bottom

1. **Data contract** (`schema.py`). A versioned Pandera schema validates raw loan data at ingest and fails closed, so bad data is rejected at the door rather than becoming a silent wrong prediction later.
2. **Ingestion** (`data/load.py`, `data/generate.py`). Reads a real Lending Club file if present, otherwise generates a schema faithful synthetic one with a drift knob for testing. Everything is validated through the contract.
3. **Features** (`features/transforms.py`). One stateless function turns validated data into model ready features. It is imported by both training and serving, so the two cannot diverge.
4. **Training** (`models/train.py`, `models/evaluate.py`). Trains a scikit learn pipeline, scores it on ranking metrics (AUC, Gini, KS), logs everything to MLflow with a data hash and schema version, and registers the model only if it clears an AUC gate.
5. **Real time serving** (`serving/api.py`). A FastAPI service loads the model from the registry by name and scores one request at a time, guarded by a Pydantic contract at the edge.
6. **Batch serving** (`models/predict.py`). The same model scores a whole population on a schedule, the nightly prediction mode.
7. **Monitoring** (`monitoring/`). PSI and KS drift detection compares live data to the training baseline, tiers the result, and fans alerts out to Slack and Prometheus.
8. **Reproducibility** (`dvc.yaml`, `params.yaml`). DVC wires generate, train, and monitor into one pipeline that reruns only what changed.
9. **Containerization** (`docker/`). A lean image serves the API, and a Compose stack runs it next to an MLflow server so the model loads over the network.
10. **Kubernetes** (`k8s/`). The API runs as a Deployment behind a Service with liveness and readiness probes, and an autoscaler that scaled it from 2 to 10 pods under load.

## Architecture

```
   raw loans (real CSV or synthetic)
        |  schema.py validates at ingest (fail closed)
        v
   labeled data  ->  build_features()  ->  model ready features
        |                       (shared by train and serve)
        +-----------------+-------------------+
        v                 v                   v
     TRAIN             SERVE                MONITOR
   train.py          api.py               drift.py
   MLflow + gate     FastAPI              PSI / KS -> Slack / Prometheus
        |            predict.py (batch)        |
        +-------- model registry --------------+
                         |
   reproduced by DVC: generate -> train -> monitor
   shipped by Docker, run on Kubernetes with autoscaling
```

## Quickstart

```powershell
# 1. isolated environment, then install the package + dev tools
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. tests
pytest

# 3. train (MLflow registry + drift baseline, gated)
python -m quant_risk.models.train
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://localhost:5000

# 4. serve real time
uvicorn quant_risk.serving.api:app --port 8000           # http://localhost:8000/docs

# 5. batch score
python -m quant_risk.models.predict --n 20000

# 6. drift monitor (force an alert)
python -m quant_risk.monitoring.run_monitor --drift 1.2

# 7. reproducible pipeline
dvc repro
dvc metrics show

# 8. containerize
docker build -t credit-pd-api -f docker/Dockerfile.api .
docker compose -f docker/docker-compose.yml up -d

# 9. kubernetes
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/hpa.yaml
```

Drop a real Lending Club CSV into `data/raw/` and everything switches from synthetic to real with no code change.

## Project structure

```
quant-risk-mlops/
├── pyproject.toml            installable package + tool config
├── requirements-serve.txt    lean serving dependencies (for the image)
├── dvc.yaml / params.yaml    reproducible pipeline + tracked knobs
├── conf/config.yaml          runtime settings (seed, paths, MLflow)
├── src/quant_risk/
│   ├── schema.py             the data contract
│   ├── config.py             central configuration
│   ├── data/                 generate.py, load.py
│   ├── features/transforms.py shared feature engineering
│   ├── models/               evaluate.py, train.py, predict.py
│   ├── serving/              schema.py, api.py
│   └── monitoring/           drift.py, alerts.py, run_monitor.py
├── tests/                    14 tests (contract, parity, metrics, api, drift)
├── docker/                   Dockerfile.api, docker-compose.yml
├── k8s/                      mlflow.yaml, api.yaml, hpa.yaml, loadgen.yaml
└── docs/                     per technology deep dives
```

## Status

| Layer | State |
|---|---|
| Data contract, config, ingestion | built |
| Shared features, training, registry, AUC gate | built |
| Test suite (14 tests) | built |
| Real time API + batch scoring | built |
| Drift monitoring + Slack / Prometheus alerts | built |
| Reproducible pipeline (DVC) | built |
| Docker image + Compose stack | built |
| Kubernetes Deployment, Service, probes, autoscaling | built |
| CI, Spark / Kafka scaling, Airflow, S3, Azure sibling | roadmap |

## Design principles

1. Guard the boundary, not the model. The contract validates raw data at ingest and fails closed.
2. Compute features in exactly one place. One shared function makes train and serve skew impossible.
3. A model is a depreciating asset. Monitoring is a first class feature.
4. Reproducibility is non negotiable. Seeds, a data hash, the schema version, and a DVC pipeline.
5. Separate the platform from the payload. The model is swappable, the platform is not.

## Documentation

Per technology deep dives live in `docs/`. Each page covers what the technology is, what it does, the commands and basic syntax, how it fits this project and where it comes alive, plus links to the official docs.

1. Kubernetes: [docs/kubernetes.md](docs/kubernetes.md)
2. Docker: [docs/docker.md](docs/docker.md)
3. MLflow: [docs/mlflow.md](docs/mlflow.md)
4. DVC: docs/dvc.md (planned)
5. Monitoring, PSI and KS: docs/monitoring.md (planned)

Planned cross cutting notes: why an isolated virtual environment, how reproducibility is enforced end to end, how technologies are deliberately swappable (the model, the data source, the orchestrator) and why, and what is intentionally kept out of version control via `.gitignore` (data, the MLflow store, local artifacts) versus committed (code, pipeline definitions, `dvc.lock`, metrics).

## Tech stack

Python, pandas, numpy, scipy. Pandera for the data contract. scikit learn for the model. MLflow for tracking and the registry. FastAPI, Pydantic, and uvicorn for serving. PSI and KS, with optional Evidently, for drift. Slack and Prometheus for alerting. DVC for the pipeline. Docker and Kubernetes for packaging and serving. pytest and ruff for quality.

## Roadmap

CI with GitHub Actions (ruff, pytest, image build on push), PySpark medallion ETL and a Kafka streaming scorer for scale, an Airflow nightly batch DAG, an S3 raw data store, and the Azure native sibling project.
