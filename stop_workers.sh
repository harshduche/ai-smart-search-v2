#!/bin/bash
# Stop all workers and model server

cd /home/deair/video-rag

echo "Stopping workers and model server..."

# Stop workers
if [ -f logs/workers.pid ]; then
    while read PID; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping worker PID: $PID"
            kill $PID
        fi
    done < logs/workers.pid
    rm logs/workers.pid
fi

# Stop any remaining worker.py processes
pkill -f "python3 worker.py" && echo "Stopped remaining workers"

# Stop model server
if [ -f logs/model_server.pid ]; then
    PID=$(cat logs/model_server.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping model server PID: $PID"
        kill $PID
        rm logs/model_server.pid
    fi
fi

# Stop any remaining model server processes
pkill -f "model_server_batched.py" && echo "Stopped model server"

sleep 2
echo "✓ All stopped"
