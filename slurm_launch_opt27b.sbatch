# still have problem in srun i will fix with eng / metwalli


#!/bin/bash
# SLURM launcher for FSDP + LoRA fine-tuning of OPT-2.7B on CPU cluster.
# - One process per node; shards model across N nodes (min 10 for 8GB RAM fit).
# - Uses --export=DATA_FILE for tiny/medium; defaults to medium.
# - Logs to ./logs; times in timing_rank*.log; model in ./opt27b_fsdp_lora_cpu.

mkdir -p logs  # Create logs dir FIRST for SLURM outputs

#SBATCH -J opt27b_fsdp
#SBATCH -p torch
#SBATCH -N 10  # Min 10 nodes for 5.4GB model in 8GB RAM; override with --nodes
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00  # Longer for medium dataset
#SBATCH --exclusive
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Paths
WORKDIR="$HOME/project/labs/FineTuning_Distilgpt2"
VENV_ACT="/opt/llamaenv/bin/activate"
mkdir -p "$WORKDIR"

echo "[SLURM Start] In $(pwd), WORKDIR=$WORKDIR"

# Rendezvous vars
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=12345
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE=${SLURM_NNODES}
export GLOO_SOCKET_TIMEOUT=1200

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export DATA_FILE="${DATA_FILE:-/data/medium_openwebtext.txt}"

echo "[SLURM] Nodes=$WORLD_SIZE, Data=$DATA_FILE, Master=$MASTER_ADDR:$MASTER_PORT"

# Checks
if [ ! -d "$WORKDIR" ]; then echo "ERROR: WORKDIR missing!"; exit 1; fi
if [ ! -f "$VENV_ACT" ]; then echo "ERROR: Venv missing!"; exit 1; fi
scontrol show hostnames "$SLURM_JOB_NODELIST"

# Propagate DATA_FILE
export DATA_FILE

# Launch with full env sharing
srun --export=ALL --ntasks=$WORLD_SIZE --ntasks-per-node=1 --cpu-bind=cores --chdir="$WORKDIR" bash -lc '
  set -euo pipefail
  mkdir -p logs

  echo "[Task on $(hostname)] Starting in $(pwd)"

  # Activate venv
  if [ -f "'"$VENV_ACT"'" ]; then
    set +u; source "'"$VENV_ACT"'"; set -u
    echo "[Task] Venv: $(which python)"
  else
    echo "ERROR: No venv!"; exit 1
  fi

  # Env from venv_env.sh
  export PYTHONPATH=/opt/llamaenv/lib/python3.11/site-packages
  export DS_BUILD_OPS=0
  export TORCH_DISTRIBUTED_BACKEND=gloo
  export ACCELERATE_DISABLE=false
  export CUDA_VISIBLE_DEVICES=""
  export USE_CPU_ONLY=true
  export HF_DATASETS_CACHE=~/.cache/huggingface/datasets_$HOSTNAME
  export RANK=${SLURM_NODEID}
  export LOCAL_RANK=0
  export TOKENIZERS_PARALLELISM=false
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export PYTHONUNBUFFERED=1
  export FSDP_CPU_RAM_EFFICIENT_LOADING=1

  echo "[Task] Data: $DATA_FILE on $(hostname)"

  # Check data
  if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data missing on $(hostname)!"
    ls -la /data/ 2>/dev/null || echo "No /data?"
    exit 1
  fi
  echo "[Task] Data OK: $(du -h "$DATA_FILE" | cut -f1)"

  # Torch check
  python -c "import torch; print(f'[Task] Torch {torch.__version__} ready!')"

  # Run with logs
  python -u train_opt27b_fsdp_lora.py 2>&1 | tee "logs/train_rank${RANK}.log"
'