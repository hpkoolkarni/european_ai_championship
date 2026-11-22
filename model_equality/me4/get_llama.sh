#!/bin/bash
# login node (internet): download public Llama-3.2 1B/3B Instruct mirrors, weights + tokenizer only
cd "$(dirname "$0")"
for m in unsloth/Llama-3.2-1B-Instruct unsloth/Llama-3.2-3B-Instruct; do
  venv/bin/python -c "from huggingface_hub import snapshot_download as s; p=s('$m', allow_patterns=['*.json','*.safetensors','tokenizer*']); print('$m ->', p)" 2>&1 | grep -v Fetching
done
echo "llama download done $(date +%H:%M)"
