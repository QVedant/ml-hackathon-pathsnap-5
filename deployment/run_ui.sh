#!/usr/bin/env bash
# Laptop 2: serve the evaluator UI. Then open http://localhost:5500 and enter laptop 1's IP.
cd "$(dirname "$0")/../ui"
python3 -m http.server 5500
