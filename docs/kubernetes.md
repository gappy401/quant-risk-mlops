# Kubernetes — Containerized, Autoscaling Serving

> Deep-dive page for the **Kubernetes** layer of `quant-risk-mlops`. The README gives the one-paragraph summary of each technology; this page is the detail. Sibling pages (Docker, MLflow, DVC, Monitoring, Spark) follow the same shape.

## What this layer does

It runs the credit-PD scoring API on Kubernetes as a **load-balanced, self-healing, autoscaling** service, with a containerized MLflow registry alongside it. The model is pulled from the in-cluster MLflow server at pod startup; readiness gating ensures traffic only reaches pods that have actually loaded the model; and a HorizontalPodAutoscaler adds and removes replicas as CPU load changes.

Run locally on Docker Desktop's built-in (kind-based) Kubernetes.

## In-cluster topology

```
                  Service: credit-pd-api (ClusterIP :8000)
                              │  load-balances
        ┌──────────┬──────────┼──────────┬───────────┐
        ▼          ▼          ▼          ▼           ▼
      pod        pod        pod        pod   ...   (2 → 10, via HPA)
   credit-pd-api Deployment  (each: readiness /ready, liveness /health)
        │
        │  MLFLOW_TRACKING_URI = http://mlflow:5000   (Service DNS)
        ▼
   Service: mlflow (ClusterIP :5000)  ──►  mlflow Deployment (registry + artifacts)

   HorizontalPodAutoscaler  ──watches CPU via metrics-server──►  scales credit-pd-api
```

## The objects

### `k8s/mlflow.yaml` — registry as a Service
A single-replica Deployment running `mlflow server` (reusing the API image, which already has MLflow), fronted by a `mlflow` Service so other pods reach it at `http://mlflow:5000`. Two settings matter: `MLFLOW_SERVER_ALLOWED_HOSTS: "*"` (see Gotchas) and `runAsUser: 0` so the server can write its volume. Storage is `emptyDir` — ephemeral; for persistence across pod restarts this becomes a PersistentVolumeClaim.

### `k8s/api.yaml` — the Deployment + Service
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-pd-api
  labels: { app: credit-pd-api }
spec:
  replicas: 2
  selector:
    matchLabels: { app: credit-pd-api }
  template:
    metadata:
      labels: { app: credit-pd-api }
    spec:
      containers:
        - name: api
          image: credit-pd-api
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow:5000"
          resources:
            requests: { cpu: "100m", memory: "256Mi" }
            limits:   { cpu: "500m", memory: "512Mi" }
          readinessProbe:
            httpGet: { path: /ready, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 10
            periodSeconds: 15
---
apiVersion: v1
kind: Service
metadata:
  name: credit-pd-api
spec:
  selector: { app: credit-pd-api }
  ports:
    - { port: 8000, targetPort: 8000 }
```

### `k8s/hpa.yaml` — the autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: credit-pd-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: credit-pd-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50
```

## Key concepts

- **Deployment = declarative desired state.** You declare "N replicas of this image"; Kubernetes continuously reconciles reality to match — replacing dead pods, rescheduling on node loss.
- **Service = stable address over ephemeral pods.** Pods' IPs churn; the Service is a fixed virtual IP + DNS name that load-balances across pods matching its `selector`. This is why the API reaches MLflow as `http://mlflow:5000`.
- **Readiness vs liveness probes.** `readinessProbe → /ready`: a pod joins the load-balancer **only** once the model is loaded (200), so traffic never hits a cold pod — zero-downtime by construction. `livenessProbe → /health`: a wedged pod is restarted automatically — self-healing.
- **Resource requests power the HPA.** The HPA computes utilization as current CPU ÷ the pod's CPU `request` (`100m`). No request → no HPA.
- **Rolling restart = the reload mechanism.** `kubectl rollout restart` brings up new pods, waits for readiness, then retires the old — how a new model version is deployed without downtime.
- **HPA control loop.** Every ~15s it reads average CPU (from metrics-server) and targets 50% utilization: over → add pods (to `maxReplicas`), under → remove (to `minReplicas`). Scale-up is fast; scale-down has a ~5-minute stabilization window to prevent flapping.

## Deploy & run

```powershell
# cluster (Docker Desktop -> Settings -> Kubernetes -> Enable)
kubectl get nodes                                  # STATUS Ready

# registry
kubectl apply -f k8s/mlflow.yaml

# API
kubectl apply -f k8s/api.yaml                      # pods start 0/1 (no model yet)

# populate the in-cluster registry, then roll the API to load it
kubectl port-forward svc/mlflow 5001:5000          # leave running
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
python -m quant_risk.models.train                  # in a second shell
kubectl rollout restart deploy/credit-pd-api       # pods become 1/1

# reach it
kubectl port-forward svc/credit-pd-api 8000:8000   # http://localhost:8000/docs

# autoscaling
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl top pods                                   # confirm CPU/MEM numbers appear
kubectl apply -f k8s/hpa.yaml
```

## Result — autoscaling under load

A throwaway Job (`k8s/loadgen.yaml`, 4 parallel pods hammering `/score`) drove CPU well past the 50% target. The HPA scaled the Deployment from 2 to the maximum of 10 replicas:

```
NAME                                                REFERENCE                  TARGETS         MINPODS   MAXPODS   REPLICAS
horizontalpodautoscaler.autoscaling/credit-pd-api   Deployment/credit-pd-api   cpu: 214%/50%   2         10        10

# 10 credit-pd-api pods, all 1/1 Running, plus 4 loadgen pods driving the load
pod/credit-pd-api-7cff9df576-6mkzd   1/1   Running
pod/credit-pd-api-7cff9df576-8vt2g   1/1   Running
pod/credit-pd-api-7cff9df576-8wj7d   1/1   Running
pod/credit-pd-api-7cff9df576-bw8c2   1/1   Running
pod/credit-pd-api-7cff9df576-cv7jr   1/1   Running
pod/credit-pd-api-7cff9df576-d2rvv   1/1   Running
pod/credit-pd-api-7cff9df576-jk2np   1/1   Running
pod/credit-pd-api-7cff9df576-k9rt4   1/1   Running
pod/credit-pd-api-7cff9df576-wws55   1/1   Running
pod/credit-pd-api-7cff9df576-z9cl7   1/1   Running
pod/mlflow-764f57c489-pvq5n          1/1   Running
```

CPU at **214% of the 50% target** → the loop requested more replicas until it hit `maxReplicas: 10`. When the load Job finished, utilization fell and the Deployment scaled back to 2 after the stabilization window.

## Gotchas solved (the real debugging)

- **MLflow DNS-rebinding 403.** The API's request to `http://mlflow:5000` was rejected with `Invalid Host header - possible DNS rebinding attack detected`. MLflow's server validates the `Host` header and only trusts localhost/private patterns by default; the in-network hostname `mlflow:5000` isn't trusted. Fix: `MLFLOW_SERVER_ALLOWED_HOSTS` (set to `*` for local dev, or an explicit allow-list in production). A security default colliding with service-name networking.
- **No metrics-server.** Docker Desktop's kind-based cluster ships without it, so the HPA showed `<unknown>` utilization. Installed it, plus `--kubelet-insecure-tls` because kind's kubelet serving certs aren't signed by the cluster CA.
- **The in-cluster registry is separate.** The MLflow server in the cluster is a fresh, empty registry — the model from the Docker-Compose MLflow does not carry over. It must be trained into via `port-forward`, after which a `rollout restart` makes the pods load it.

## Production differences (next)

This is a local single-node cluster. A real deployment would add: a managed cluster (AKS/EKS/GKE), an **Ingress** + TLS instead of `port-forward`, a **PersistentVolumeClaim** for MLflow, **Secrets** for credentials (not `*` host-allow), resource tuning, and GitOps-style applies (Argo CD / Flux) instead of `kubectl apply` by hand.
