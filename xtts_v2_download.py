#!/usr/bin/env python3

from huggingface_hub import snapshot_download

# Repository details
repo_id = 'coqui/XTTS-v2'
local_dir = './xtts_v2_weights'  # Change this to your desired local directory

# Download the entire repository
snapshot_download(repo_id, local_dir=local_dir)

print(f"Repository {repo_id} has been downloaded to {local_dir}")
