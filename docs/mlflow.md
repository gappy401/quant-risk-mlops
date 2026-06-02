# MLflow — Tracking and the Model Registry

> Rundown page for the **MLflow** layer of `quant-risk-mlops`, in the same shape as the other technology pages.

## What MLflow is

MLflow is an open source platform for managing the machine learning lifecycle. In this project it does three jobs: it records what happened during training (tracking), it saves the trained model in a standard format (models), and it keeps a named, versioned catalog of models that serving code can load from (the model registry). A tracking server hosts all of this as a network service.

## What it does (the parts used here)

1. **Tracking**: every training run logs its parameters, metrics, and output files (artifacts), so runs are comparable and reproducible.
2. **Models**: a model is saved in a standard layout using a flavor such as `mlflow.sklearn`, so it can be loaded back the same way regardless of who trained it.
3. **Model Registry**: a logged model can be registered under a name like `credit_pd`, which then accumulates numbered versions.
4. **Tracking Server**: a standalone service that stores all of the above and serves it over HTTP, so a trainer and a separate API can share one source of truth.

## Core concepts

1. **Experiment and run**: a run is one training execution, grouped under a named experiment.
2. **Params, metrics, artifacts**: inputs you log (params), measured outcomes (metrics), and files such as the model itself (artifacts).
3. **Flavor**: the framework specific save format, here `mlflow.sklearn`.
4. **Registered model and versions**: a name (`credit_pd`) with an ordered list of versions.
5. **Model URI**: how you ask for a model, for example `models:/credit_pd/latest` or by a specific version or alias.
6. **Tracking URI**: where the client sends data, for example `sqlite:///mlflow.db` locally or `http://mlflow:5000` for a server.
7. **Backend store vs artifact store**: the backend store holds metadata (params, metrics, the registry), while the artifact store holds the actual model files.

## The commands and API

1. `mlflow ui --backend-store-uri sqlite:///mlflow.db` opens the local browser UI.
2. `mlflow server --host 0.0.0.0 --port 5000 ... --serve-artifacts` runs the tracking server as a service.
3. In Python during training:

```python
import mlflow, mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")   # where to log
with mlflow.start_run():
    mlflow.log_params({"seed": 42, "schema_version": "1.1.0"})
    mlflow.log_metrics({"auc": 0.72, "gini": 0.44, "ks": 0.32})
    mlflow.sklearn.log_model(model, name="model",
                             registered_model_name="credit_pd")  # log + register
```

4. In Python when serving:

```python
model = mlflow.sklearn.load_model("models:/credit_pd/latest")   # load by name
```

## How it fits this project, and where it comes alive

Training (`models/train.py`) logs the run's parameters including a data hash and the schema version, logs the ranking metrics (AUC, Gini, KS), and logs the model itself. It registers the model under the name `credit_pd` only if it clears the AUC gate, so a weak model is recorded for inspection but never promoted. Serving (`serving/api.py`) and batch scoring (`models/predict.py`) then load the model by name from the registry, which means the serving code never needs to know how or when the model was trained. That decoupling is the registry's whole purpose: it is the versioned handoff point between the people who train and the system that serves.

Where it comes alive is that handoff. In the Docker and Kubernetes stacks, MLflow runs as its own server that both the trainer and the API talk to over the network. When training wrote `credit_pd` into the in cluster server and the API pods loaded `models:/credit_pd/latest` and passed their readiness check, the registry was doing exactly its job: one trained artifact, versioned and named, served by many replicas that found it by name. The registry is also where one real gotcha lived, the server's host header protection rejecting the in network name until it was allowed, which is documented on the Kubernetes page.

## Documentation links

1. MLflow documentation home: https://mlflow.org/docs/latest/
2. Tracking: https://mlflow.org/docs/latest/ml/tracking/
3. Model Registry workflow: https://mlflow.org/docs/latest/ml/model-registry/workflow/
