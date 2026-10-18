# Heart Disease Risk Prediction – End-to-End MLOps

[![ML CI/CD pipeline](https://github.com/naaz-bits/MLOPS-2025AF05094-Naaz-assignment1/actions/workflows/ci.yml/badge.svg)](https://github.com/naaz-bits/MLOPS-2025AF05094-Naaz-assignment1/actions/workflows/ci.yml)

MLOps (AIMLCZG523) Assignment 1. Predicts presence of heart disease from 13 clinical
features (UCI Heart Disease, Cleveland subset) and serves the model as a monitored,
containerised API on Kubernetes.

**Student:** Naaz (2025AF05094)

## Submission

| Deliverable | Location |
|-------------|----------|
| Final report (PDF) | [`report/`](report/) |
| Pipeline video | _add link_ |
| Screenshots (50 + architecture) | [`screenshots/`](screenshots/) |
| CI/CD runs | [GitHub Actions](https://github.com/naaz-bits/MLOPS-2025AF05094-Naaz-assignment1/actions) |
| Cleaned dataset / download script | `data/processed/heart_clean.csv` / `src/data.py` |
| EDA, training, inference | `notebooks/01_eda.ipynb`, `src/train.py`, `notebooks/02_inference.ipynb` |
| Deployment manifests | [`k8s/`](k8s/) |
| API access | local Kubernetes, no public URL — see [Deploy to Kubernetes](#deploy-to-kubernetes) |

## Results

Three models tuned with `GridSearchCV` (5-fold stratified CV, ROC-AUC) on an 80/20 stratified split.
The winner is picked on **CV ROC-AUC**, not the 61-patient test set.

| Model | CV ROC-AUC | Test ROC-AUC | Test accuracy | Test recall |
|-------|-----------:|-------------:|--------------:|------------:|
| Logistic Regression | 0.9039 | 0.968 | 0.885 | 0.929 |
| **Random Forest** (selected) | **0.9047** | 0.950 | 0.885 | 0.929 |
| Gradient Boosting | 0.8741 | 0.961 | 0.902 | 0.929 |

Random Forest: 100 trees, max depth 5, min samples per leaf 5. Full metrics in
`reports/model_comparison.csv` and in MLflow (experiment `heart-disease-classification`).

## Architecture

![Architecture](screenshots/00_architecture.png)

<details><summary>Mermaid source</summary>

```mermaid
flowchart LR
    UCI[(UCI repository)] -->|src/data.py| RAW[data/raw] --> CLEAN[data/processed/heart_clean.csv]
    CLEAN -->|src/eda.py| FIG[reports/figures]
    CLEAN -->|src/train.py<br/>GridSearchCV + 5-fold CV| MLF[(MLflow<br/>params · metrics · plots · models)]
    MLF -->|best CV ROC-AUC| MODEL[models/model.joblib<br/>preprocess + model Pipeline]
    MODEL --> IMG[Docker image<br/>FastAPI + uvicorn]
    subgraph GHA[GitHub Actions]
      L[lint] --> T[pytest] --> TR[train + quality gate] --> D[docker build + smoke test]
    end
    IMG --> K8S
    subgraph K8S[Kubernetes · namespace heart-disease]
      LB[LoadBalancer :8080] --> P1[heart-api pod]
      LB --> P2[heart-api pod]
      PROM[Prometheus] -. scrape /metrics .-> P1 & P2
      GRAF[Grafana] --> PROM
    end
    USER((client)) -->|POST /predict| LB
```

</details>

## Repository layout

| Path | Purpose |
|------|---------|
| `src/data.py` | download + clean dataset (`python -m src.data`) |
| `src/eda.py` | EDA figures → `reports/figures/` |
| `src/features.py` | `ColumnTransformer` preprocessing (impute, scale, one-hot) |
| `src/train.py` | tune + compare LogReg / RandomForest / GradientBoosting, MLflow logging, saves best model |
| `api/main.py` | FastAPI service: `/predict`, `/health`, `/info`, `/metrics` |
| `tests/` | pytest unit tests for data, pipeline and API |
| `notebooks/` | EDA and inference notebooks |
| `.github/workflows/ci.yml` | lint → test → train → docker build/smoke test |
| `Dockerfile`, `docker-compose.yml` | container + local monitoring stack |
| `k8s/` | Deployment, LoadBalancer Service, Prometheus, Grafana, `deploy.sh` |
| `monitoring/` | Prometheus config, Grafana datasource + dashboard |
| `screenshots/` | evidence for the report |

## Setup (clean machine)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data      # 1. download + clean  -> data/processed/heart_clean.csv
python -m src.eda       # 2. EDA figures       -> reports/figures/
python -m src.train     # 3. train + MLflow    -> models/model.joblib, mlflow.db
pytest                  # 4. unit tests
ruff check src api tests

mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

## Run the API

**Locally**
```bash
uvicorn api.main:app --reload            # http://127.0.0.1:8000/docs
```

**Docker**
```bash
docker build -t heart-api:1.0 .
docker run --rm -p 8000:8000 heart-api:1.0
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @sample_request.json
```

Example response:
```json
{"model": "random_forest",
 "results": [{"prediction": 1, "label": "disease", "confidence": 0.9754, "probability_disease": 0.9754},
             {"prediction": 0, "label": "no_disease", "confidence": 0.93, "probability_disease": 0.07}]}
```

**Docker Compose with monitoring:** `docker compose up --build` starts the API (:8000),
Prometheus (:9090) and Grafana (:3000).

## Deploy to Kubernetes

**Docker Desktop** — the LoadBalancer maps straight to `localhost`:

1. Docker Desktop → Settings → Kubernetes → *Enable Kubernetes* → Apply.
2. `kubectl config use-context docker-desktop`
3. `./k8s/deploy.sh`

**Minikube** (used for the submitted deployment, on the Prayogshala Linux VM) — the image must be
loaded into the cluster, and services are reached with port-forward (or `minikube tunnel`):

```bash
minikube start --driver=docker --cpus=2 --memory=4096
kubectl config use-context minikube
docker build -t heart-api:1.0 . && minikube image load heart-api:1.0
./k8s/deploy.sh
kubectl -n heart-disease port-forward svc/heart-api  8080:8080 &
kubectl -n heart-disease port-forward svc/prometheus 9090:9090 &
kubectl -n heart-disease port-forward svc/grafana    3000:3000 &
curl -s localhost:8080/health
curl -s -X POST localhost:8080/predict -H 'content-type: application/json' -d @sample_request.json
```

Either way:

| Service | URL |
|---------|-----|
| API (LoadBalancer) | http://localhost:8080/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 → *Heart Disease API – Monitoring* |

Generate traffic for the dashboard: `./scripts/send_traffic.sh http://localhost:8080 300`

View request logs: `kubectl -n heart-disease logs -l app=heart-api -f`

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | one patient object or a list → prediction, label, confidence |
| GET | `/health` | liveness / readiness |
| GET | `/info` | model name, hyper-parameters, metrics |
| GET | `/metrics` | Prometheus metrics |

Input fields: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
(`ca` and `thal` optional → imputed). Invalid values return HTTP 422.
