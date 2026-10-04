#!/usr/bin/env bash
# One-time setup of the DMPEL track on xulab (2x RTX 5090, Blackwell sm_120), everything under /mnt/data/users/bbouri.
# The system disk (/) has ~14 GB free and is shared, so NOTHING may be written there: every cache, temp dir and
# config dir is redirected below. No sudo is used.
#   nohup bash scripts/xulab_setup.sh > /mnt/data/users/bbouri/logs/setup.log 2>&1 &
set -euo pipefail
XB=/mnt/data/users/bbouri
CL=$XB/cl
H=$XB/CL-benchmark
mkdir -p $XB/tmp $XB/logs $XB/cache/{xdg,config,pip,hf,torch,conda_pkgs,mpl,wandb} $CL/{third_party,checkpoints,config/dmpel,outputs} $CL/data/libero/libero_10

cat > $XB/env_x.sh <<'EOF'
# source /mnt/data/users/bbouri/env_x.sh : keep every write off the system disk
export XB=/mnt/data/users/bbouri
export CL_ROOT=$XB/cl
export CLB_ROOT=$XB/CL-benchmark
export CLB_DISK_PATH=$XB
export CLB_ROOT_MIN_GB=5
export CLB_GPU_SAMPLE_S=60
export CLB_NICE="nice -n 10"
export CLB_HARNESS_ENV=$CL_ROOT/env_dmpel.sh
export TMPDIR=$XB/tmp
export XDG_CACHE_HOME=$XB/cache/xdg XDG_CONFIG_HOME=$XB/cache/config
export PIP_CACHE_DIR=$XB/cache/pip HF_HOME=$XB/cache/hf TORCH_HOME=$XB/cache/torch
export CONDA_PKGS_DIRS=$XB/cache/conda_pkgs MPLCONFIGDIR=$XB/cache/mpl
export WANDB_DIR=$CLB_ROOT/wandb WANDB_CONFIG_DIR=$XB/cache/config/wandb WANDB_CACHE_DIR=$XB/cache/wandb
export LIBERO_CONFIG_PATH=$CL_ROOT/config/dmpel
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# DMPEL saves pickled config objects in its checkpoints; torch>=2.6 refuses them unless this is set
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
EOF
source $XB/env_x.sh

echo "== [1/7] Miniforge under $XB"
if [ ! -x $XB/miniforge3/bin/conda ]; then
  curl -sSL -o $XB/tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash $XB/tmp/Miniforge3.sh -b -p $XB/miniforge3
  rm -f $XB/tmp/Miniforge3.sh
fi
source $XB/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q "^dmpel " || conda create -y -n dmpel python=3.10
conda activate dmpel
cat > $CL/env_dmpel.sh <<EOF
source $XB/env_x.sh
source $XB/miniforge3/etc/profile.d/conda.sh
conda activate dmpel
EOF

echo "== [2/7] PyTorch for Blackwell (cu128) and pinned dependencies"
pip install -q --upgrade "pip<25.3" "setuptools==69.5.1" wheel
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q "numpy==1.23.5" "hydra-core==1.2.0" "easydict==1.9" "transformers==4.21.1" "opencv-python==4.6.0.66" \
  "einops==0.4.1" "thop" "bddl==1.0.1" "future" "matplotlib" "cloudpickle==2.1.0" "gym==0.25.2" "timm==1.0.15" \
  "scikit-learn==1.3.2" "mujoco==2.3.7" "h5py" "tensorboard" "pyyaml" "tqdm" "imageio" "pytest" \
  "wandb>=0.17" "huggingface_hub>=0.24,<1.0"
pip install -q "robosuite==1.4.1" "robomimic==0.2.0"
pip install -q "setuptools==69.5.1"   # DMPEL imports pkg_resources.packaging (removed in newer setuptools)

echo "== [3/7] DMPEL code at the released commit"
if [ ! -d $CL/third_party/DMPEL ]; then git clone -q https://github.com/HarryLui98/DMPEL.git $CL/third_party/DMPEL; fi
git -C $CL/third_party/DMPEL checkout -q b1abe28
pip install -q -e $CL/third_party/DMPEL --config-settings editable_mode=compat
python $CONDA_PREFIX/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py > /dev/null
D=$CL/third_party/DMPEL/libero/libero
cat > $CL/config/dmpel/config.yaml <<EOF
benchmark_root: $D
bddl_files: $D/bddl_files
init_states: $D/init_files
datasets: $CL/data/libero
assets: $D/assets
EOF

echo "== [4/7] Harness patches on DMPEL"
cd $H
for s in patch_dmpel_datasets patch_dmpel_protocol patch_dmpel_grad_accum patch_dmpel_ddp_ckpt; do python scripts/$s.py; done

echo "== [5/7] Checkpoint, CLIP, and the two LIBERO-Long task files used by the micro-run"
python - <<'EOF'
import os, re, urllib.request
from huggingface_hub import snapshot_download, hf_hub_download
CL = os.environ["CL_ROOT"]
snapshot_download("leiyuheng/DMPEL", local_dir=f"{CL}/checkpoints/dmpel_pretrain")
snapshot_download("openai/clip-vit-base-patch16", local_dir=f"{CL}/checkpoints/hf/clip-vit-base-patch16",
                  allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
t = open(f"{CL}/third_party/DMPEL/libero/libero/benchmark/libero_suite_task_map.py").read()
seg = t[t.index('"libero_10"'):]
names = re.findall(r'"([A-Z_0-9a-z]+)"', seg)[1:3]   # task order 0 and 1
for n in names:
    p = hf_hub_download("yifengzhu-hf/LIBERO-datasets", f"libero_10/{n}_demo.hdf5", repo_type="dataset", local_dir=f"{CL}/data/libero")
    print("data:", p, round(os.path.getsize(p) / 1e9, 2), "GB")
EOF

echo "== [6/7] Verification (CPU/GPU-light; one GPU, no rendering)"
CUDA_VISIBLE_DEVICES=1 python - <<'EOF' 2>&1 | grep -viE "warn|deprecat"
import os, torch, h5py
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("arch list:", torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list(), "this torch build has no Blackwell kernels"
x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item(); print("matmul ok on", torch.cuda.get_device_name(0))
ck = torch.load(os.environ["CL_ROOT"] + "/checkpoints/dmpel_pretrain/multitask_model_ep10.pth", map_location="cpu")
print("DMPEL checkpoint loads; keys:", list(ck.keys())[:3])
import timm; m = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True); print("timm CLIP ok")
from transformers import CLIPModel; CLIPModel.from_pretrained(os.environ["CL_ROOT"] + "/checkpoints/hf/clip-vit-base-patch16"); print("transformers CLIP ok")
import libero.lifelong.algos, libero.lifelong.models, libero.lifelong.metric  # noqa
from libero.libero import benchmark, get_libero_path
b = benchmark.get_benchmark_dict()["libero_10"](); print("libero_10 tasks:", b.get_num_tasks(), "| data dir:", get_libero_path("datasets"))
for i in (0, 1):
    p = os.path.join(get_libero_path("datasets"), b.get_task_demonstration(i)); f = h5py.File(p, "r"); print("task", i, os.path.basename(p)[:60], "demos", len(f["data"]))
print("VERIFY OK")
EOF
pip check 2>&1 | head -5 || true

echo "== [7/7] Footprint"
du -sh $XB/miniforge3 $CL $XB/cache 2>/dev/null
df -h / $XB | tail -2
echo "XULAB SETUP DONE"
