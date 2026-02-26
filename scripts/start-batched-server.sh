#!/bin/bash
# Start Batched Model Server optimized for 12GB VRAM

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Starting Batched Model Server (12GB VRAM Optimized)      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • GPU VRAM: 12GB"
echo "  • Batch window: 50ms (collects concurrent requests)"
echo "  • Max batch size: 6 requests in parallel"
echo "  • Expected VRAM: 8-10GB peak (leaves 2-4GB headroom)"
echo "  • Port: 8001"
echo ""

# Load environment (skip comments and empty lines)
set -a
source <(grep -v '^#' .env.batched-server | grep -v '^[[:space:]]*$')
set +a

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Start batched model server
echo "Starting server..."
python model_server_batched.py
