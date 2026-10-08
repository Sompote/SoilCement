#!/bin/sh
# Runs the examples of the README from the repository root.
set -e
echo "== no trial mix: zero-shot law"
soilcement predict --ll 74 --pi 37 --query examples/singapore_query.csv --out pred_zero.csv
echo "== three trial mixes, automatic model choice (TabICL), Singapore clay held out of the corpus"
soilcement predict --ll 74 --pi 37 --trials examples/singapore_trials.csv --query examples/singapore_query.csv \
    --exclude-deposit "Singapore marine clay" --out pred_tabicl.csv
echo "== binder content for 800 kPa at 120 % water content and 28 days"
soilcement design --ll 74 --pi 37 --trials examples/singapore_trials.csv --target 800 --water 120 --curing 28 \
    --exclude-deposit "Singapore marine clay"
