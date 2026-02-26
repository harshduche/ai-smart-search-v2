#!/bin/bash
# Start Model Server with logging

cd /home/deair/video-rag

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if already running
if lsof -i :8001 >/dev/null 2>&1; then
    echo "❌ Model server already running on port 8001"
    exit 1
fi

# Start model server in background
echo "Starting Model Server on port 8001..."
nohup python3 model_server_batched.py > logs/model_server.log 2>&1 &
PID=$!

echo "Model server started with PID: $PID"
echo $PID > logs/model_server.pid

# Wait a bit and check if it started successfully
sleep 5

if ps -p $PID > /dev/null; then
    echo "✓ Model server is running"
    echo "  Logs: tail -f logs/model_server.log"
    echo "  Health: curl http://localhost:8001/health"
else
    echo "❌ Model server failed to start. Check logs/model_server.log"
    exit 1
fi
