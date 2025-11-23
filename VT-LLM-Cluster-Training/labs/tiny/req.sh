# req.sh
export OFFLINE_ROOT="$HOME/offline_repo_py311"
mkdir -p "$OFFLINE_ROOT/pkgs" "$OFFLINE_ROOT/wheels"

# Clean any previous conflicting NumPy build (safe to skip, but avoids mix-ups)
rm -rf "$OFFLINE_ROOT/pkgs/numpy" "$OFFLINE_ROOT/pkgs/numpy-"*.dist-info 2>/dev/null || true

cat > "$OFFLINE_ROOT/requirements_local.txt" <<'REQ'
torch==2.3.1+cpu
transformers==4.41.2
tokenizers==0.19.1
datasets==2.19.0
# ↓ Use NumPy 1.x to avoid datasets 2.19.0 + NumPy 2.0 formatting crash
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
typing_extensions==4.12.2
sentencepiece==0.1.99
accelerate==0.31.0
safetensors==0.4.3
tqdm==4.66.4
REQ

# If the login node HAS internet:
python3.11 -m pip install -t "$OFFLINE_ROOT/pkgs" \
  --only-binary=:all: \
  --upgrade --force-reinstall \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$OFFLINE_ROOT/requirements_local.txt"

# If NO internet on login node:
python3.11 -m pip download -d "$OFFLINE_ROOT/wheels" \
  --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$OFFLINE_ROOT/requirements_local.txt"

python3.11 -m pip install -t "$OFFLINE_ROOT/pkgs" \
  --no-index --find-links "$OFFLINE_ROOT/wheels" \
  --upgrade --force-reinstall \
  -r "$OFFLINE_ROOT/requirements_local.txt"
