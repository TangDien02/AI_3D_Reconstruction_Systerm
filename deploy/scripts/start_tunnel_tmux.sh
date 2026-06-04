#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-tunnel}"
TARGET_URL="${TARGET_URL:-http://127.0.0.1:8010}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed."
  echo "Install from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  exit 1
fi

tmux new-session -d -s "$SESSION" "cloudflared tunnel --url '$TARGET_URL'"
echo "Started Cloudflare tunnel tmux session '$SESSION'."
echo "Attach: tmux attach -t $SESSION"
echo "Look for the https://....trycloudflare.com URL in the tunnel log."
