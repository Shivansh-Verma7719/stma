#!/bin/bash

SESSION_NAME="sp500_pipeline"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$PROJECT_ROOT/pipelines"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  echo "Creating new tmux session: $SESSION_NAME"
  # Create a new detached session
  tmux new-session -d -s $SESSION_NAME -c "$PIPELINE_DIR"

  # Send commands to the session
  # 1. Activate environment
  tmux send-keys -t $SESSION_NAME "source env/bin/activate" C-m
  
  # 2. Run the pipeline
  tmux send-keys -t $SESSION_NAME "python sp500_media_pipeline.py" C-m
  
  echo "Pipeline started in tmux session '$SESSION_NAME'."
else
  echo "Session '$SESSION_NAME' already exists."
fi

echo ""
echo "To attach to the session, run:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from the session, press: Ctrl+B then D"
