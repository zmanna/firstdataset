#!/bin/sh

# Source this file to activate the project venv and Kaggle credentials together.
PROJECT_DIR="/Users/mannz/Desktop/polymer degredation/firstdataset"

export KAGGLE_CONFIG_DIR="$PROJECT_DIR/.kaggle"
export KAGGLE_API_TOKEN="$PROJECT_DIR/.kaggle/access_token"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

. "$PROJECT_DIR/.venv/bin/activate"
