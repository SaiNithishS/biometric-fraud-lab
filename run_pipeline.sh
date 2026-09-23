#!/usr/bin/env bash
set -e
mkdir -p data/raw data/processed reports
python src/generate_data.py
python src/detect.py
python src/visualise.py
python src/report.py
