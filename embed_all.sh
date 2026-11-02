#!/bin/sh
cd "$(dirname "$0")"
for job in "scifact base" "nfcorpus small" "nfcorpus base" "fiqa small" "fiqa base"; do
  .venv/bin/python embed.py $job 2>&1 | grep -E "^(scifact|nfcorpus|fiqa) " 
done
echo ALL_DONE
