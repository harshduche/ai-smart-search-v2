#!/bin/bash

# Quick start script for Docker deployment
set -e

echo "=========================================="
echo "Visual Search Engine - Docker Launcher"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo "Please edit .env if needed and run this script again."
    echo ""
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    USE_GPU=true
else
    echo "⚠️  No NVIDIA GPU detected. Will use CPU mode."
    USE_GPU=false
fi

echo ""
echo "Select mode:"
echo "1) Production (GPU)"
echo "2) Production (CPU)"
echo "3) Development (GPU with hot-reload)"
echo "4) Stop all services"
echo "5) View logs"
echo "6) Rebuild containers"
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "Starting in Production mode (GPU)..."
        docker-compose up -d
        ;;
    2)
        echo ""
        echo "Starting in Production mode (CPU)..."
        # Modify docker-compose to use CPU version
        echo "⚠️  Please manually edit docker-compose.yml to use api-cpu service"
        echo "Comment out 'api' service and uncomment 'api-cpu' service"
        exit 1
        ;;
    3)
        echo ""
        echo "Starting in Development mode (GPU with hot-reload)..."
        docker-compose -f docker-compose.dev.yml up
        ;;
    4)
        echo ""
        echo "Stopping all services..."
        docker-compose down
        docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
        echo "✓ Services stopped"
        exit 0
        ;;
    5)
        echo ""
        echo "Showing logs (Ctrl+C to exit)..."
        docker-compose logs -f
        exit 0
        ;;
    6)
        echo ""
        echo "Rebuilding containers..."
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Services starting..."
echo "=========================================="
echo ""
echo "Waiting for services to be healthy..."
sleep 5

# Check service health
echo ""
echo "Checking Qdrant..."
if curl -s http://localhost:6333/ > /dev/null 2>&1; then
    echo "✓ Qdrant is running"
else
    echo "⚠️  Qdrant not responding yet (may still be starting)"
fi

echo ""
echo "Checking API..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API is running"
else
    echo "⚠️  API not responding yet (models may be loading)"
    echo "   Run: docker-compose logs -f api"
fi

echo ""
echo "=========================================="
echo "✓ Services launched!"
echo "=========================================="
echo ""
echo "Access the application:"
echo "  • Dashboard:  http://localhost:8000/static/index.html"
echo "  • API Docs:   http://localhost:8000/docs"
echo "  • Qdrant:     http://localhost:6333/dashboard"
echo ""
echo "Useful commands:"
echo "  • View logs:        docker-compose logs -f"
echo "  • Stop services:    docker-compose down"
echo "  • Restart:          docker-compose restart"
echo "  • Enter container:  docker-compose exec api bash"
echo ""
echo "For more information, see docker-README.md"
echo ""
