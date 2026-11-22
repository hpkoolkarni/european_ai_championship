#!/bin/bash
# login node (internet): download the practice base family SmolLM2 135M / 360M / 1.7B Instruct
cd "$(dirname "$0")"
for m in HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct; do
  venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('$m', allow_patterns=['*.json','*.safetensors','tokenizer*','merges.txt','vocab.json']); print('$m ok')" 2>&1 | grep -v Fetching
done
echo "smol download done $(date +%H:%M)"
