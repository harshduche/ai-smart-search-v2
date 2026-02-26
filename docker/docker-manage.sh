#!/bin/bash

# Video-RAG Docker Management Script
# Usage: ./docker-manage.sh [command]

set -e

COMPOSE_FILE="docker-compose.yml"
DEV_COMPOSE_FILE="docker-compose.dev.yml"
WORKER_COMPOSE_FILE="docker-compose.worker.yml"
MODEL_SERVER_COMPOSE_FILE="docker-compose.model-server.yml"
WITH_MODEL_SERVER_COMPOSE_FILE="docker-compose.with-model-server.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

show_help() {
    cat << EOF
Video-RAG Docker Management Script

Usage: ./docker-manage.sh [command]

Commands:
  start               Start all services (production)
  start-dev           Start all services (development with hot reload)
  start-worker        Start standalone worker only (connects to external services)
  start-model-server  Start model server only (for distributed workers)
  start-with-model    Start full stack with centralized model server
  stop                Stop all services
  stop-worker         Stop standalone worker
  stop-model-server   Stop model server
  restart             Restart all services
  restart-worker      Restart only the worker
  logs                View logs from all services
  logs-worker         View worker logs only
  logs-model-server   View model server logs
  status              Show status of all services
  status-worker       Show status of standalone worker
  scale [N]           Scale workers to N instances (e.g., scale 3)
  scale-worker [N]    Scale standalone workers to N instances
  build               Rebuild all images
  clean               Stop and remove all containers, networks, volumes
  publish-job         Publish a test job to the queue
  health              Check health of all services
  stats               Show resource usage stats
  gpu-check           Check if GPU is available
  shell-worker        Open shell in worker container
  shell-api           Open shell in API container

Examples:
  ./docker-manage.sh start            # Start production stack
  ./docker-manage.sh start-dev        # Start development stack
  ./docker-manage.sh start-worker     # Start standalone worker only
  ./docker-manage.sh logs-worker      # View worker logs
  ./docker-manage.sh scale 3          # Run 3 worker instances
  ./docker-manage.sh scale-worker 5   # Run 5 standalone workers
  ./docker-manage.sh restart-worker   # Restart worker only
  ./docker-manage.sh health           # Check service health

EOF
}

check_env() {
    if [ ! -f .env ]; then
        print_warning ".env file not found!"
        print_info "Creating from .env.example..."
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success ".env created"
            print_warning "Please edit .env with your configuration before starting services"
            exit 1
        else
            print_error ".env.example not found!"
            exit 1
        fi
    fi
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "GPU available"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
            return 0
        fi
    fi
    print_warning "No GPU detected - will use CPU mode"
    return 1
}

# Commands
cmd_start() {
    print_info "Starting Video-RAG services (production)..."
    check_env
    docker compose -f $COMPOSE_FILE up -d
    print_success "Services started"
    cmd_status
}

cmd_start_dev() {
    print_info "Starting Video-RAG services (development)..."
    check_env
    docker compose -f $DEV_COMPOSE_FILE up -d
    print_success "Development services started"
    cmd_status
}

cmd_start_worker() {
    print_info "Starting standalone worker (connects to external services)..."
    check_env
    print_warning "Make sure RabbitMQ and Qdrant are accessible via .env configuration"
    docker compose -f $WORKER_COMPOSE_FILE up -d
    print_success "Standalone worker started"
    cmd_status_worker
}

cmd_start_model_server() {
    print_info "Starting model server (for distributed workers)..."
    check_env
    docker compose -f $MODEL_SERVER_COMPOSE_FILE up -d
    print_success "Model server started"
    sleep 5
    cmd_health_model_server
}

cmd_start_with_model() {
    print_info "Starting full stack with centralized model server..."
    check_env
    print_info "Architecture: 1 GPU model server + CPU-only workers"
    docker compose -f $WITH_MODEL_SERVER_COMPOSE_FILE up -d
    print_success "Full stack with model server started"
    docker compose -f $WITH_MODEL_SERVER_COMPOSE_FILE ps
}

cmd_stop_model_server() {
    print_info "Stopping model server..."
    docker compose -f $MODEL_SERVER_COMPOSE_FILE down
    print_success "Model server stopped"
}

cmd_logs_model_server() {
    print_info "Viewing model server logs (Ctrl+C to exit)..."
    docker compose -f $MODEL_SERVER_COMPOSE_FILE logs -f
}

cmd_health_model_server() {
    print_info "Checking model server health..."
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        print_success "Model server is healthy (http://localhost:8001)"
        curl -s http://localhost:8001/health | python3 -m json.tool
    else
        print_error "Model server is not responding (http://localhost:8001)"
    fi
}

cmd_stop_worker() {
    print_info "Stopping standalone worker..."
    docker compose -f $WORKER_COMPOSE_FILE down
    print_success "Standalone worker stopped"
}

cmd_status_worker() {
    print_info "Standalone Worker Status:"
    docker compose -f $WORKER_COMPOSE_FILE ps
}

cmd_scale_worker() {
    if [ -z "$1" ]; then
        print_error "Please specify number of workers (e.g., ./docker-manage.sh scale-worker 3)"
        exit 1
    fi
    print_info "Scaling standalone workers to $1 instances..."
    docker compose -f $WORKER_COMPOSE_FILE up -d --scale worker=$1
    print_success "Standalone workers scaled to $1"
}

cmd_stop() {
    print_info "Stopping all services..."
    docker compose -f $COMPOSE_FILE down
    docker compose -f $DEV_COMPOSE_FILE down 2>/dev/null || true
    print_success "Services stopped"
}

cmd_restart() {
    print_info "Restarting all services..."
    docker compose -f $COMPOSE_FILE restart
    print_success "Services restarted"
}

cmd_restart_worker() {
    print_info "Restarting worker..."
    docker compose -f $COMPOSE_FILE restart worker
    print_success "Worker restarted"
}

cmd_logs() {
    print_info "Viewing logs (Ctrl+C to exit)..."
    docker compose -f $COMPOSE_FILE logs -f
}

cmd_logs_worker() {
    print_info "Viewing worker logs (Ctrl+C to exit)..."
    docker compose -f $COMPOSE_FILE logs -f worker
}

cmd_status() {
    print_info "Service Status:"
    docker compose -f $COMPOSE_FILE ps
}

cmd_scale() {
    if [ -z "$1" ]; then
        print_error "Please specify number of workers (e.g., ./docker-manage.sh scale 3)"
        exit 1
    fi
    print_info "Scaling workers to $1 instances..."
    docker compose -f $COMPOSE_FILE up -d --scale worker=$1
    print_success "Workers scaled to $1"
}

cmd_build() {
    print_info "Building Docker images..."
    docker compose -f $COMPOSE_FILE build --no-cache
    print_success "Images built"
}

cmd_clean() {
    print_warning "This will remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        docker compose -f $COMPOSE_FILE down -v
        docker compose -f $DEV_COMPOSE_FILE down -v 2>/dev/null || true
        print_success "Cleanup complete"
    else
        print_info "Cancelled"
    fi
}

cmd_publish_job() {
    print_info "Publishing test job..."

    if [ -f "examples/sample_jobs.json" ]; then
        python3 scripts/publish_embedding_job.py --batch-file examples/sample_jobs.json
    else
        print_warning "examples/sample_jobs.json not found"
        print_info "Publishing single test job..."
        python3 scripts/publish_embedding_job.py \
            --video-url "https://example.com/test-video.mp4" \
            --metadata '{"organization_id": "test-org", "flight_id": "test-flight"}'
    fi

    print_success "Job published"
}

cmd_health() {
    print_info "Checking service health..."
    echo ""

    # Check API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy (http://localhost:8000)"
    else
        print_error "API is not responding (http://localhost:8000)"
    fi

    # Check RabbitMQ
    if curl -s http://localhost:15672 > /dev/null 2>&1; then
        print_success "RabbitMQ is healthy (http://localhost:15672)"
    else
        print_error "RabbitMQ is not responding (http://localhost:15672)"
    fi

    # Check Qdrant
    if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
        print_success "Qdrant is healthy (http://localhost:6333)"
    else
        print_error "Qdrant is not responding (http://localhost:6333)"
    fi

    # Check Worker
    if docker compose ps worker | grep -q "Up"; then
        print_success "Worker is running"
    else
        print_error "Worker is not running"
    fi

    echo ""
    print_info "Service URLs:"
    echo "  API:          http://localhost:8000"
    echo "  API Docs:     http://localhost:8000/docs"
    echo "  RabbitMQ UI:  http://localhost:15672 (guest/guest)"
    echo "  Qdrant:       http://localhost:6333"
}

cmd_stats() {
    print_info "Resource Usage:"
    docker stats --no-stream
}

cmd_gpu_check() {
    print_info "Checking GPU availability..."
    check_gpu
    echo ""
    print_info "Testing GPU access in Docker..."
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>/dev/null; then
        print_success "GPU is accessible from Docker"
    else
        print_error "GPU is not accessible from Docker"
        print_info "Make sure nvidia-container-toolkit is installed"
    fi
}

cmd_shell_worker() {
    print_info "Opening shell in worker container..."
    docker compose exec worker /bin/bash
}

cmd_shell_api() {
    print_info "Opening shell in API container..."
    docker compose exec api /bin/bash
}

# Main command dispatcher
case "${1:-}" in
    start)
        cmd_start
        ;;
    start-dev)
        cmd_start_dev
        ;;
    start-worker)
        cmd_start_worker
        ;;
    start-model-server)
        cmd_start_model_server
        ;;
    start-with-model)
        cmd_start_with_model
        ;;
    stop)
        cmd_stop
        ;;
    stop-worker)
        cmd_stop_worker
        ;;
    stop-model-server)
        cmd_stop_model_server
        ;;
    restart)
        cmd_restart
        ;;
    restart-worker)
        cmd_restart_worker
        ;;
    logs)
        cmd_logs
        ;;
    logs-worker)
        cmd_logs_worker
        ;;
    logs-model-server)
        cmd_logs_model_server
        ;;
    status)
        cmd_status
        ;;
    status-worker)
        cmd_status_worker
        ;;
    scale)
        cmd_scale "$2"
        ;;
    scale-worker)
        cmd_scale_worker "$2"
        ;;
    build)
        cmd_build
        ;;
    clean)
        cmd_clean
        ;;
    publish-job)
        cmd_publish_job
        ;;
    health)
        cmd_health
        ;;
    stats)
        cmd_stats
        ;;
    gpu-check)
        cmd_gpu_check
        ;;
    shell-worker)
        cmd_shell_worker
        ;;
    shell-api)
        cmd_shell_api
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac
