#!/bin/bash

# Script to pull this folder from the remote server to your local machine
# This script should be run from your LOCAL machine, not on the remote server
#
# Usage (from your local machine):
#   ssh user@remote-host 'cat /home/deair/video-rag/scripts/scp_pull.sh' | bash -s /local/destination/path
#
# Or copy this script to your local machine and run:
#   ./scp_pull.sh user@host:/home/deair/video-rag /local/destination/path

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo "Usage: $0 user@host:/remote/path /local/destination/path"
    echo "Example: $0 user@example.com:/home/deair/video-rag ~/projects/video-rag"
    exit 1
fi

REMOTE_SOURCE=$1
LOCAL_DEST=$2

echo -e "${GREEN}Pulling video-rag project from: ${REMOTE_SOURCE}${NC}"
echo -e "${GREEN}To local destination: ${LOCAL_DEST}${NC}"
echo -e "${YELLOW}Excluding files based on remote .gitignore${NC}"
echo ""

# Create local destination if it doesn't exist
mkdir -p "$LOCAL_DEST"

# Extract host and path from REMOTE_SOURCE
if [[ $REMOTE_SOURCE =~ ^(.+):(.+)$ ]]; then
    REMOTE_HOST="${BASH_REMATCH[1]}"
    REMOTE_PATH="${BASH_REMATCH[2]}"
else
    echo -e "${RED}Error: Invalid remote source format${NC}"
    echo "Expected format: user@host:/path"
    exit 1
fi

echo "Fetching .gitignore from remote..."
TEMP_GITIGNORE=$(mktemp)
scp "$REMOTE_HOST:$REMOTE_PATH/.gitignore" "$TEMP_GITIGNORE" 2>/dev/null || {
    echo -e "${YELLOW}Warning: Could not fetch .gitignore, copying all files${NC}"
    rsync -avz --progress --exclude '.git/' -e ssh "$REMOTE_SOURCE/" "$LOCAL_DEST/"
    rm -f "$TEMP_GITIGNORE"
    exit 0
}

# Use rsync to pull from remote to local
rsync -avz --progress \
    --exclude-from="$TEMP_GITIGNORE" \
    --exclude '.git/' \
    --exclude '*.log' \
    -e ssh \
    "$REMOTE_SOURCE/" "$LOCAL_DEST/"

# Clean up temp file
rm -f "$TEMP_GITIGNORE"

echo ""
echo -e "${GREEN}✓ Pull completed successfully!${NC}"
echo -e "Files copied to: ${LOCAL_DEST}"
