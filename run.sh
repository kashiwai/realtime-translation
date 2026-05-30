#!/usr/bin/env bash
# APIキー(.env)を読み込み、venvを有効化して通訳を起動するヘルパー。
# 例: ./run.sh --mode a   /  ./run.sh --list-devices
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && source .env
[ -d .venv ] && source .venv/bin/activate

exec python -u run.py "$@"
