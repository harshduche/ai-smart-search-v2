#!/bin/bash
# Reset RabbitMQ queue for fresh start

QUEUE_NAME="${1:-embedding.jobs}"

echo "Deleting queue: $QUEUE_NAME"

# If using Docker
if docker ps | grep -q rabbitmq; then
    docker exec video-rag-rabbitmq rabbitmqctl delete_queue "$QUEUE_NAME"
    echo "✓ Queue deleted successfully"
else
    # If RabbitMQ is running locally
    sudo rabbitmqctl delete_queue "$QUEUE_NAME"
    echo "✓ Queue deleted successfully"
fi

echo ""
echo "Queue will be recreated with proper configuration on next job publish."
