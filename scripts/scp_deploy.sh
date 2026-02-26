#!/bin/bash

# Script to copy this folder to a remote server via SCP/rsync while respecting .gitignore
# Usage: ./scripts/scp_deploy.sh user@host:/path/to/destination

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory (where the script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if destination is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No destination provided${NC}"
    echo "Usage: $0 user@host:/path/to/destination"
    echo "Example: $0 user@example.com:/home/user/video-rag"
    exit 1
fi

DESTINATION=$1

echo -e "${GREEN}Copying video-rag project to: ${DESTINATION}${NC}"
echo -e "${YELLOW}Excluding files based on .gitignore${NC}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check if .gitignore exists
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}Warning: .gitignore not found, copying all files${NC}"
    rsync -avz --progress -e ssh . "$DESTINATION"
    exit 0
fi

# Use rsync with exclude-from option
# rsync is more efficient than scp and can handle .gitignore patterns
# We also exclude .git directory by default
rsync -avz --progress \
    --exclude-from='.gitignore' \
    --exclude '.git/' \
    --exclude 'scripts/scp_deploy.sh.bak' \
    -e ssh \
    . "$DESTINATION"

echo ""
echo -e "${GREEN}✓ Copy completed successfully!${NC}"
echo -e "Files copied to: ${DESTINATION}"
