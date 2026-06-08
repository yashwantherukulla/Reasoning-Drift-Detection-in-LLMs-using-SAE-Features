#!/bin/bash
# Wrapper script for run_all.py for Linux/Mac users
# Usage: ./scripts/run_all.sh --pilot
#        ./scripts/run_all.sh --full

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--pilot | --full]"
    exit 1
fi

uv run scripts/run_all.py "$@"
