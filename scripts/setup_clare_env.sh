#!/usr/bin/env bash
# Phase 1: conda env `clare` for the CLARE codebase (learnsyslab/clare) + gym-libero + assets.
#   nohup bash scripts/setup_clare_env.sh > logs/setup_clare_env.log 2>&1 &
# gym_libero is required by CLARE's LIBERO env factory but is not listed in the CLARE repo;
# ZhangYi1999/gym-libero (CLARE co-author) matches its env ids, 256 px obs and _init_state_id API.
set -euo pipefail
CL=/usr1/home/bbouri/continual-learning
T=$CL/third_party
source $CL/miniforge3/etc/profile.d/conda.sh
export PIP_CACHE_DIR=$CL/cache/pip HF_HOME=$CL/cache/hf HF_LEROBOT_HOME=$CL/cache/lerobot

conda env list | grep -q "^clare " || conda create -y -n clare python=3.10
conda activate clare
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e $T/clare/peft_lsy
pip install -e $T/clare/lerobot_lsy
if [ ! -d $T/gym-libero ]; then GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/ZhangYi1999/gym-libero.git $T/gym-libero; fi
pip install -e $T/gym-libero
pip install "wandb>=0.17"
# Pins (Phase 1, see docs/DECISIONS.md): peft_lsy needs transformers.HybridCache (gone in 5.x);
# versions follow CLARE's lerobot pixi.lock. MuJoCo must be 3.x here: with 2.3.7 gym_libero renders broken
# materials (verified 2026-10-01, docs/DECISIONS.md) and CLARE scores 0%.
pip install "transformers==4.56.2" "diffusers==0.35.1" "mujoco==3.3.0" "huggingface-hub>=0.34,<1.0"

# assets: base checkpoint (pretrained on LIBERO-90, CLARE's regenerated 256 px data) + LIBERO-10 tasks 0-1
python - <<'EOF'
from huggingface_hub import snapshot_download
import os
root = "/usr1/home/bbouri/continual-learning/checkpoints/clare"
os.makedirs(root, exist_ok=True)
p = snapshot_download("continuallearning/dit_flow_mt_libero_90_pretrain", local_dir=f"{root}/dit_flow_mt_libero_90_pretrain")
print("checkpoint:", p)
for t in (0, 1):
    p = snapshot_download(f"continuallearning/libero_10_image_task_{t}", repo_type="dataset")
    print("dataset:", p)
EOF
python -c "import torch, lerobot, peft, gym_libero; print('torch', torch.__version__, torch.cuda.is_available(), '| lerobot ok | peft', peft.__version__, '| gym_libero ok')"
for r in clare gym-libero; do echo "$r $(git -C $T/$r rev-parse --short HEAD)"; done
echo "CLARE ENV DONE"
