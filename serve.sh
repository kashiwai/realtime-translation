#!/usr/bin/env bash
# APIキー(.env)を読み込み、venvを有効化してローカルWeb UIサーバを起動する。
# ブラウザで http://127.0.0.1:8765 を開いて設定・開始・停止する。
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && source .env
[ -d .venv ] && source .venv/bin/activate

exec python -u server.py "$@"
