# Copy video-rag to Local Machine

Since you're SSH'd into the remote machine, here are the ways to copy files to your local machine:

## Option 1: One-liner from your local terminal (Recommended)

Open a **new terminal on your LOCAL machine** (not this SSH session) and run:

```bash
rsync -avz --progress \
    --exclude-from=<(ssh user@remote-host 'cat /home/deair/video-rag/.gitignore') \
    --exclude '.git/' \
    -e ssh \
    user@remote-host:/home/deair/video-rag/ \
    ~/destination-folder/
```

Replace:
- `user@remote-host` with your SSH connection details
- `~/destination-folder/` with where you want the files locally

## Option 2: Using the pull script

From your **LOCAL machine**, run:

```bash
# First, copy the script to your local machine
scp user@remote-host:/home/deair/video-rag/scripts/scp_pull.sh ~/scp_pull.sh

# Make it executable
chmod +x ~/scp_pull.sh

# Run it
~/scp_pull.sh user@remote-host:/home/deair/video-rag ~/destination-folder
```

## Option 3: Simple rsync without .gitignore filtering

If you just want a quick copy and don't mind getting all files:

```bash
rsync -avz --progress --exclude '.git/' \
    user@remote-host:/home/deair/video-rag/ \
    ~/destination-folder/
```

## Option 4: Create a git archive (if it's a git repo)

From this SSH session, create a tar archive excluding git-ignored files:

```bash
cd /home/deair/video-rag
git archive --format=tar.gz -o /tmp/video-rag.tar.gz HEAD
```

Then from your local machine:
```bash
scp user@remote-host:/tmp/video-rag.tar.gz ~/
tar -xzf ~/video-rag.tar.gz -C ~/destination-folder/
```

---

**Note:** Replace `user@remote-host` with your actual SSH connection details (e.g., `deair@192.168.1.100` or `deair@example.com`)
