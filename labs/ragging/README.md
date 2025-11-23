# Retrieval-Augmented Generation Demo

This directory hosts a lightweight, CPU-only RAG example built for teaching purposes.

## Files

- `rag_example.py` – end-to-end script that fits a TF–IDF retriever over a small AG News subset and generates answers with FLAN-T5-Small.
- `side_shell` – annotated shell transcript showing how to pre-cache models, allocate a node, run queries and monitor jobs.

## Quickstart

1. **Pre-cache models and dataset** (once):
   ```bash
   python - <<'PY'
   import os, datasets, transformers
   os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_rag")
   datasets.load_dataset("ag_news", split="train[:2000]", cache_dir=f"{os.environ['HF_HOME']}/ds")
   transformers.AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=f"{os.environ['HF_HOME']}/tok")
   transformers.AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=f"{os.environ['HF_HOME']}/gen")
   print("Done pre-caching for RAG.")
   PY
   ```
2. **Allocate a node and activate your environment**:
   ```bash
   salloc -N1 -n1 -c2 --mem=2G -p parallel --time=00:20:00 --pty bash
   source ~/llamaenv_local/bin/activate
   cd ~/project
   export HF_HOME=$HOME/.cache/hf_rag
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```
3. **Single-query demo**:
   ```bash
   python labs/ragging/rag_example.py --dry_run --query "What is deep learning?"
   ```
4. **Parallel batch demo** (two processes on one node):
   ```bash
   cat > labs/ragging/queries.txt <<'EOF'
   What happened in the sports world?
   Which company announced a new product?
   How did the stock market perform?
   Describe a political event mentioned.
   EOF

   export MASTER_ADDR=$(hostname)
   export MASTER_PORT=$((20000 + RANDOM % 10000))
   torchrun --nproc_per_node=2 labs/ragging/rag_example.py \
     --dry_run --queries_file labs/ragging/queries.txt --k 3 --max_new_tokens 64
   ```
5. **Monitoring** (run in another terminal):
   ```bash
   watch -n2 "sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveCPU,MaxVMSize"
   watch -n2 "ps -u $USER -o pid,pcpu,pmem,etime,cmd | grep python | grep -v grep"
   watch -n5 'squeue -u $USER -o "%.9i %.2t %.10M %.18R %.12C %j"'
   ```

These steps mirror the `side_shell` script while providing a concise reference for the RAG lab.
