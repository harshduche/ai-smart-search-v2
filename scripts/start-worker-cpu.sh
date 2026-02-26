#!/bin/bash
# Start a CPU Worker that connects to Model Server

echo "Starting CPU Worker (connects to Model Server)..."
echo "This worker uses NO GPU - it forwards embedding requests to the model server"
echo ""

# Load environment
export $(grep -v '^#' .env.worker | xargs)

# Verify model server is running
echo "Checking model server at $MODEL_SERVER_URL..."
if curl -s "${MODEL_SERVER_URL}/health" > /dev/null 2>&1; then
    echo "✓ Model server is running"
else
    echo "✗ Model server is NOT running at $MODEL_SERVER_URL"
    echo "  Please start the model server first: ./start-model-server.sh"
    exit 1
fi

echo ""
echo "Starting worker..."
python worker.py
