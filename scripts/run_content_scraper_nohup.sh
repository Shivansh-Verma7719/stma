#!/bin/bash

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/content_scraper_nohup.log"
mkdir -p "$PROJECT_ROOT/logs"

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Run the pipeline with nohup
echo "Starting Content Scraper V2 with nohup in background..."
echo "Logs: $LOG_FILE"
echo "Monitor: tail -f logs/content_scraper_nohup.log"

# -u for unbuffered output so logs appear immediately
nohup python -u pipelines/content_scraper_pipeline_v2.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Content Scraper running with PID: $PID"
