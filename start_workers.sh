#!/bin/bash
# Start multiple workers with remote embeddings

cd /home/deair/video-rag

# Number of workers to start (default: 4)
NUM_WORKERS=${1:-4}

# Check if model server is running
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "❌ Model server not running on port 8001"
    echo "   Run: ./start_model_server.sh first"
    exit 1
fi

echo "Starting $NUM_WORKERS workers with remote embeddings..."
echo "Model Server: http://localhost:8001"
echo ""

# Create logs directory
mkdir -p logs

# Export environment for remote embeddings
export USE_REMOTE_EMBEDDINGS=true
export MODEL_SERVER_URL=http://localhost:8001
export MODEL_SERVER_TIMEOUT=300

# Start workers
for i in $(seq 1 $NUM_WORKERS); do
    echo "Starting worker $i..."
    nohup python3 worker.py > logs/worker_$i.log 2>&1 &
    WORKER_PID=$!
    echo "  Worker $i PID: $WORKER_PID"
    echo $WORKER_PID >> logs/workers.pid
    sleep 1
done

echo ""
echo "✓ Started $NUM_WORKERS workers"
echo "  View logs: tail -f logs/worker_*.log"
echo "  Stop all: ./stop_workers.sh"
