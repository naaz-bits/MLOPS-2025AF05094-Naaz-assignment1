#!/usr/bin/env bash
# Build the image and deploy API + Prometheus + Grafana to the current kube context.
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -t heart-api:1.0 .

kubectl apply -f k8s/00-namespace.yaml
for pair in "grafana-datasources:monitoring/grafana/provisioning/datasources" \
            "grafana-dashboard-provider:monitoring/grafana/provisioning/dashboards" \
            "grafana-dashboards:monitoring/grafana/dashboards"; do
  kubectl -n heart-disease create configmap "${pair%%:*}" --from-file="${pair#*:}" \
    --dry-run=client -o yaml | kubectl apply -f -
done
kubectl apply -f k8s/

kubectl -n heart-disease rollout restart deployment/heart-api
kubectl -n heart-disease rollout status deployment/heart-api --timeout=120s
kubectl -n heart-disease get pods,svc
echo
echo "API:        http://localhost:8080/docs"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000   (dashboard: Heart Disease API – Monitoring)"
