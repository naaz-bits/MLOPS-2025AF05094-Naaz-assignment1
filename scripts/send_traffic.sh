#!/usr/bin/env bash
# Generate traffic so the Grafana dashboard has something to show.
#   ./scripts/send_traffic.sh [base_url] [n_requests]
URL=${1:-http://localhost:8080}
N=${2:-200}
cd "$(dirname "$0")/.."
for i in $(seq 1 "$N"); do
  curl -s -o /dev/null -X POST "$URL/predict" -H 'content-type: application/json' -d @sample_request.json
  # an occasional invalid request so error/4xx panels are populated
  if (( i % 15 == 0 )); then
    curl -s -o /dev/null -X POST "$URL/predict" -H 'content-type: application/json' -d '{"age": -1}'
  fi
done
echo "sent $N requests to $URL"
