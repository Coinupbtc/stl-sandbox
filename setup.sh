#!/usr/bin/env bash
# Alias: same as ./install.sh (one-command name visitors expect)
set -euo pipefail
cd "$(dirname "$0")"
exec ./install.sh "$@"
