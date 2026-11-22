#!/bin/bash
# one-off setup on a JURECA login node (has internet): separate venv + model downloads, all under me4/ (no API calls)
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p $R/me4/hf
cd $R/me4
export HF_HOME=$R/me4/hf PIP_CACHE_DIR=$R/me4/pipcache
echo "start $(date +%H:%M)"
$R/.venv/bin/python -c "import sys; print(sys.version)"
if [ ! -x venv/bin/python ]; then
  ( $R/.venv/bin/python -m venv venv 2>/dev/null || python3 -m venv venv )
fi
venv/bin/python -m pip install -q --upgrade pip
venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121
venv/bin/python -m pip install -q "transformers>=4.44" accelerate safetensors huggingface_hub
venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| transformers', transformers.__version__)"
echo "packages done $(date +%H:%M)"
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct; do
  venv/bin/python -c "from huggingface_hub import snapshot_download as s; p=s('$m', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','merges.txt','vocab.json']); print('$m ->', p)"
done
du -sh $R/me4/hf
echo "all done $(date +%H:%M)"
