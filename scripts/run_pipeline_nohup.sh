#!/bin/bash

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/pipeline_nohup.log"
mkdir -p "$PROJECT_ROOT/logs"

# Ensure we're in the project root
cd "$PROJECT_ROOT"

echo "Initializing Conda..."

# Activate Environment using the user-provided dynamic method
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: 'conda' command not found in PATH."
    exit 1
fi

conda activate specialenv || { echo "Error: Failed to activate 'specialenv'"; exit 1; }

# Run the pipeline with nohup
echo "Starting pipeline with nohup in background..."
echo "Logs: $LOG_FILE"
echo "Monitor: tail -f logs/pipeline_nohup.log"

# -u for unbuffered output so logs appear immediately
nohup python -u pipelines/sp500_media_pipeline.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Pipeline running with PID: $PID"
