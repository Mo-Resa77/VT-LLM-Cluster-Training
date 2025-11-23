#!/usr/bin/env python3
"""
DDP inference for the tiny classifier.
Prints a single global line on rank 0 that the evaluator parses:

  [RANK 0] INFER global_accuracy=0.8730 global_samples_per_sec=1234.5 global_tokens_per_sec=158016.0
"""

import os, time, argparse
import torch

def dist_is_init() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()
def get_rank() -> int:
    return torch.distributed.get_rank() if dist_is_init() else 0
def get_world_size() -> int:
    return torch.distributed.get_world_size() if dist_is_init() else 1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to saved tiny_out/")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_test", type=int, default=2048)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = p.parse_args()

    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    if not dist_is_init():
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    print(f"[RANK {get_rank()}] WORLD_SIZE={get_world_size()}", flush=True)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset

    # Load
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt, torch_dtype=torch.float32)
    model.eval()

    # Data
    cache_root = os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))
    ds = load_dataset("ag_news", split=f"test[:{args.max_test}]", cache_dir=os.path.join(cache_root, "ds"))

    def encode(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=args.seq_len)
    ds = ds.map(encode, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Shard the dataset across ranks: simple round-robin
    world_size = get_world_size()
    rank = get_rank()
    indices = list(range(len(ds)))[rank::world_size]
    from torch.utils.data import Subset, DataLoader
    shard = Subset(ds, indices)
    loader = DataLoader(shard, batch_size=args.batch)

    # Inference loop
    local_correct = 0
    local_total = 0
    local_tokens = 0

    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            preds = logits.argmax(dim=-1)
            local_correct += (preds == batch["label"]).sum().item()
            bsz = preds.size(0)
            local_total += bsz
            local_tokens += bsz * args.seq_len
    t1 = time.perf_counter()
    local_time = t1 - t0

    # Reduce: correct, total, tokens -> SUM; time -> MAX (conservative throughput)
    tens = lambda x: torch.tensor([float(x)], dtype=torch.float64)
    corr = tens(local_correct); tot = tens(local_total); toks = tens(local_tokens); tm = tens(local_time)

    torch.distributed.all_reduce(corr, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(tot,  op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(toks, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(tm,   op=torch.distributed.ReduceOp.MAX)

    if rank == 0:
        correct = corr.item()
        total = max(1.0, tot.item())
        tokens = toks.item()
        wall = max(1e-9, tm.item())

        acc = correct / total
        sps = total / wall
        tps = tokens / wall
        print(f"[RANK 0] INFER global_accuracy={acc:.4f} "
              f"global_samples_per_sec={sps:.1f} global_tokens_per_sec={tps:.1f}",
              flush=True)

if __name__ == "__main__":
    main()
