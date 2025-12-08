#!/bin/bash

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/bert_nohup.log"
mkdir -p "$PROJECT_ROOT/logs"

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Run the pipeline with nohup
echo "Starting BERT Sentiment Pipeline with nohup in background..."
echo "Logs: $LOG_FILE"
echo "Monitor: tail -f logs/bert_nohup.log"

# Set HEADLESS=1 to force tqdm/simple logging in BERT.py
# -u for unbuffered output so logs appear immediately
export HEADLESS=1
nohup python -u pipelines/BERT.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "BERT Pipeline running with PID: $PID"
