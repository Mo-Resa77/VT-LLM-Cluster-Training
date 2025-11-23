cat > ~/launch_tiny.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

# ============ SETTINGS ============
PARTITION=torch
JOB_NODES=5
NPROC_PER_NODE=2
TIME=00:30:00
SALLOC_OPTS="${SALLOC_OPTS:-}"   # e.g., --exclude=hpc42,hpc27
DEBUG="${DEBUG:-0}"              # DEBUG=1 to disable kill-on-bad-exit and enable bash tracing

export PROJECT_DIR="$HOME/project"
export OFFLINE_ROOT="$HOME/offline_repo_py311"

[[ -d "$OFFLINE_ROOT/pkgs" ]] || { echo "ERROR: $OFFLINE_ROOT/pkgs missing"; exit 1; }
[[ -d "$PROJECT_DIR"       ]] || { echo "ERROR: $PROJECT_DIR missing"; exit 1; }

mkdir -p "$HOME/slurm_logs"

# ============ BOOTSTRAP UNDER SALLOC ============
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec salloc --partition="$PARTITION" --nodes="$JOB_NODES" \
              --ntasks-per-node=1 --cpus-per-task=4 --time="$TIME" \
              $SALLOC_OPTS \
              bash -lc "bash '$0' --inside"
fi

echo "[alloc] job=$SLURM_JOB_ID nodes=$SLURM_NODELIST"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export NNODES=${SLURM_NNODES:-$(scontrol show hostnames "$SLURM_NODELIST" | wc -l)}
echo "NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

SRUN_KILL_OPT="--kill-on-bad-exit=1"
[[ "$DEBUG" = "1" ]] && SRUN_KILL_OPT="" && echo "[debug] disable kill-on-bad-exit; enable tracing in srun blocks"

# ============ PREFLIGHT ============
echo "[preflight] checking visibility of PROJECT_DIR and OFFLINE_ROOT/pkgs on all nodes..."
VIS_REPORT=$(srun --export=ALL -N "$NNODES" -n "$NNODES" bash -lc '
  echo -n "$HOSTNAME "
  [[ -d "'"$PROJECT_DIR"'" ]] && printf "proj=ok " || printf "proj=missing "
  [[ -d "'"$OFFLINE_ROOT"'/pkgs" ]] && printf "pkgs=ok\n" || printf "pkgs=missing\n"
')
echo "$VIS_REPORT" | tee "$HOME/slurm_logs/preflight.$SLURM_JOB_ID.txt"

OK_CNT=$(echo "$VIS_REPORT" | grep -c 'proj=ok pkgs=ok' || true)
echo "[preflight] $OK_CNT/$NNODES nodes have both paths"

if [[ "$OK_CNT" -lt "$NNODES" ]]; then
  echo "[stage] staging project & pkgs via sbcast (to per-node /tmp)"
  TMP_TARS=$(mktemp -d)
  tar -C "$(dirname "$PROJECT_DIR")" -cf "$TMP_TARS/project.tar" "$(basename "$PROJECT_DIR")"
  tar -C "$OFFLINE_ROOT"            -cf "$TMP_TARS/pkgs.tar"     "pkgs"
  sbcast --compress --force "$TMP_TARS/project.tar" /tmp/project_$SLURM_JOB_ID.tar
  sbcast --compress --force "$TMP_TARS/pkgs.tar"    /tmp/pkgs_$SLURM_JOB_ID.tar

  srun --label --output="$HOME/slurm_logs/%x.%j.%n.%t.out" \
       --export=ALL -N "$NNODES" -n "$NNODES" bash -lc '
    [[ "'"$DEBUG"'" = "1" ]] && set -x
    set -e
    STAGE_DIR="${SLURM_TMPDIR:-/tmp/$USER/slurm_$SLURM_JOB_ID}"
    mkdir -p "$STAGE_DIR"
    tar -xf /tmp/project_'"$SLURM_JOB_ID"'.tar -C "$STAGE_DIR"
    tar -xf /tmp/pkgs_'"$SLURM_JOB_ID"'.tar    -C "$STAGE_DIR"
    echo "$HOSTNAME staged -> $STAGE_DIR"
  '
fi

# Per-node runtime roots using safe fallback
RUN_WRAP_ENV='
  RUN_TMP="${SLURM_TMPDIR:-/tmp/$USER/slurm_$SLURM_JOB_ID}"
  RUN_OFFLINE_ROOT="'"$OFFLINE_ROOT"'"
  RUN_PROJECT_DIR="'"$PROJECT_DIR"'"
  [[ -d "$RUN_PROJECT_DIR" ]]       || RUN_PROJECT_DIR="$RUN_TMP/'"$(basename "$PROJECT_DIR")"'"
  [[ -d "$RUN_OFFLINE_ROOT/pkgs" ]] || RUN_OFFLINE_ROOT="$RUN_TMP"

  export RUN_OFFLINE_ROOT RUN_PROJECT_DIR RUN_TMP
  export PYTHONPATH="$RUN_OFFLINE_ROOT/pkgs:$RUN_PROJECT_DIR:${PYTHONPATH:-}"

  # Caches & threading
  export HF_HOME="$HOME/.cache/hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export OMP_NUM_THREADS=2
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_DIST_BACKEND=gloo
  export PYTHONUNBUFFERED=1
'

# ============ 1) PER-NODE SANITY ============
srun --label --output="$HOME/slurm_logs/%x.%j.%n.%t.out" \
  --export=ALL --nodes="$NNODES" --ntasks="$NNODES" bash -lc "
  [[ \"$DEBUG\" = \"1\" ]] && set -x
  set -e
  $RUN_WRAP_ENV
  PYBIN=\$(command -v python3.11 || command -v python3 || command -v python || true)
  if [[ -z \"\$PYBIN\" ]]; then
    echo \"[sanity][\$(hostname)] ERROR: no python found\" >&2
    exit 1
  fi
  echo \"[\$HOSTNAME] RUN_PROJECT_DIR=\$RUN_PROJECT_DIR RUN_OFFLINE_ROOT=\$RUN_OFFLINE_ROOT\"
  cat >/tmp/_sanity.py << 'PY'
import os, sys, socket, site, traceback
try:
    root = os.environ['RUN_OFFLINE_ROOT']
    proj = os.environ['RUN_PROJECT_DIR']
    site.addsitedir(os.path.join(root, 'pkgs'))
    sys.path.insert(0, proj)
    import torch, transformers, numpy, datasets
    print('NODE', socket.gethostname(), 'OK -> PY', sys.version.split()[0],
          'torch', torch.__version__, 'tfm', transformers.__version__,
          'numpy', numpy.__version__, 'datasets', datasets.__version__,
          'root', root, flush=True)
except Exception as e:
    print('[sanity] import failed:', e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
PY
  \"\$PYBIN\" /tmp/_sanity.py && rm -f /tmp/_sanity.py
"

# ============ 2) DISTRIBUTED LAUNCH ============
srun --label --output="$HOME/slurm_logs/%x.%j.%n.%t.out" \
  --export=ALL --nodes="$NNODES" --ntasks="$NNODES" $SRUN_KILL_OPT bash -lc "
  [[ \"$DEBUG\" = \"1\" ]] && set -x
  set -e
  $RUN_WRAP_ENV
  PYBIN=\$(command -v python3.11 || command -v python3 || command -v python || true)
  if [[ -z \"\$PYBIN\" ]]; then
    echo \"[launch][\$(hostname)] ERROR: no python found\" >&2
    exit 1
  fi

  cat >/tmp/_run_wrapper.py << 'PYW'
import os, sys, site, runpy, traceback
root = os.environ['RUN_OFFLINE_ROOT']
proj = os.environ['RUN_PROJECT_DIR']
site.addsitedir(os.path.join(root, 'pkgs'))
sys.path.insert(0, proj)

# Optional NumPy>=2 patch for older datasets
try:
    import numpy as _np
    if int(_np.__version__.split('.')[0]) >= 2:
        try:
            import datasets.formatting.formatting as _fmt
            def _patched_arrow_array_to_numpy(self, array):
                return _np.asarray(array)
            _fmt.NumpyArrowExtractor._arrow_array_to_numpy = _patched_arrow_array_to_numpy
            print('[patch] Applied NumPy 2.x compatibility for datasets formatter', flush=True)
        except Exception as e:
            print('[patch] Skipped datasets patch:', e, flush=True)
except Exception:
    pass

action = os.environ.get('ACTION', 'train').lower()
default_extra = '--subset 2000 --epochs 3' if action == 'train' else ''
extra = os.environ.get('EXTRA_ARGS', default_extra)
script_rel = 'labs/tiny/infer_ddp.py' if action == 'infer' else 'labs/tiny/train_tiny.py'
script_path = os.path.join(proj, script_rel)

if not os.path.exists(script_path):
    print(f'[wrapper][ERROR] missing script: {script_path}', file=sys.stderr)
    sys.exit(2)

argv = [script_path]
if extra:
    argv.extend(extra.split())
argv.extend(sys.argv[1:])
sys.argv = argv

print(f'[wrapper] ACTION={action} EXTRA_ARGS={extra}', flush=True)
try:
    runpy.run_path(script_path, run_name='__main__')
except SystemExit as se:
    raise
except Exception:
    print('[wrapper] Uncaught exception in task', file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
PYW

  echo \"torchrun: nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=\$SLURM_NODEID rdzv=$MASTER_ADDR:$MASTER_PORT\"
  \"\$PYBIN\" -m torch.distributed.run \
    --nnodes=\"$NNODES\" \
    --nproc_per_node=\"$NPROC_PER_NODE\" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\"$MASTER_ADDR:$MASTER_PORT\" \
    --rdzv_id=\"\$SLURM_JOB_ID\" \
    --node_rank=\"\$SLURM_NODEID\" \
    /tmp/_run_wrapper.py
"

echo "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE MASTER=$MASTER_ADDR:$MASTER_PORT"
BASH

chmod +x ~/launch_tiny.sh

echo For training, use: SALLOC_OPTS="--exclude=hpc42,hpc27" ACTION=train EXTRA_ARGS="--subset 2000 --epochs 1 --batch 8" bash ~/launch_tiny.sh
echo For inference, use: SALLOC_OPTS="--exclude=hpc42,hpc27" ACTION=infer EXTRA_ARGS="--ckpt tiny_out --batch 16 --max_test 2048" bash ~/launch_tiny.sh
echo For evaluation, use: python3.11 ~/eval_logs.py --job $SLURM_JOB_ID

