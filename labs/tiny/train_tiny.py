# labs/tiny/train_tiny.py
"""
Tiny (<1 M params) text classifier with HuggingFace Trainer + torchrun.
Adds scientific signals: step_ms/samples_per_sec/tokens_per_sec, epoch eval accuracy, and total runtime.
"""

import os, argparse, time
import torch
import numpy as np

# ── CPU-only safety patch ──
if not torch.cuda.is_available():
    try:
        import torch.cpu  # type: ignore
        torch.cpu.set_device = lambda *_: None
    except Exception:
        pass

from datasets import load_dataset
from transformers import (
    BertConfig, BertForSequenceClassification,
    AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
)

# ---------- rank helpers ----------
def dist_is_init() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()
def get_rank() -> int:
    return torch.distributed.get_rank() if dist_is_init() else 0
def get_world_size() -> int:
    return torch.distributed.get_world_size() if dist_is_init() else 1

# ---------- per-rank logger ----------
class PerRankLogger(TrainerCallback):
    def __init__(self, out_dir: str, use_tb: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        self.rank = get_rank()
        self.txt_path = os.path.join(out_dir, f"log.rank{self.rank}.txt")
        self._fh = open(self.txt_path, "a", buffering=1)
        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(out_dir, "tb", f"rank{self.rank}")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb = SummaryWriter(tb_dir)
            except Exception:
                self.tb = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        step = int(state.global_step) if state.global_step is not None else -1
        prefix = f"[rank {self.rank} | step {step}] "
        line = prefix + " ".join(f"{k}={v}" for k, v in logs.items())
        print(line, flush=True)
        self._fh.write(line + "\n")
        if self.tb:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def on_train_end(self, args, state, control, **kwargs):
        try: self._fh.close()
        except Exception: pass
        if self.tb:
            try: self.tb.flush(); self.tb.close()
            except Exception: pass

# ---------- step timer (scientific signals) ----------
class StepTimer(TrainerCallback):
    def __init__(self, per_device_bs: int, seq_len: int = 128):
        self.bs = per_device_bs
        self.seq_len = seq_len
        self.t0 = None
    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()
    def on_step_end(self, args, state, control, **kw):
        if self.t0 is None: return
        dt_ms = (time.perf_counter() - self.t0) * 1000.0
        ws = get_world_size()
        sps = (self.bs * ws) / (dt_ms / 1000.0)          # samples/sec (cluster)
        tps = sps * self.seq_len                          # tokens/sec (approx)
        print(f"[rank {get_rank()} | step {state.global_step}] "
              f"step_ms={dt_ms:.2f} samples_per_sec={sps:.2f} tokens_per_sec={tps:.2f}",
              flush=True)
        self.t0 = None

# ---------- helpers ----------
def build_tokenizer(cache_root: str):
    tok_name = "google/bert_uncased_L-2_H-128_A-2"
    return AutoTokenizer.from_pretrained(tok_name,
                                         cache_dir=os.path.join(cache_root, "tok"))

def tiny_config(vocab_size: int) -> BertConfig:
    return BertConfig(
        vocab_size=vocab_size, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=256, max_position_embeddings=256,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}

# ---------- main ----------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--out",    default="./tiny_out")
    p.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = p.parse_args()

    # DDP init
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    if not dist_is_init():
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    # Rank banner
    if dist_is_init():
        print(f"[RANK {get_rank()}] WORLD_SIZE={get_world_size()}", flush=True)
    else:
        print("[single process]", flush=True)

    # Caches
    cache_root = os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "hf_tiny"))
    os.makedirs(cache_root, exist_ok=True)

    # Data
    ds = load_dataset("ag_news", cache_dir=os.path.join(cache_root, "ds"))
    ds["train"] = ds["train"].select(range(args.subset))
    ds["test"]  = ds["test"].select(range(512))

    tok = build_tokenizer(cache_root)
    def tok_fn(batch):
        return tok(batch["text"], padding="max_length", truncation=True, max_length=128)
    ds_tok = ds.map(tok_fn, batched=True).rename_column("label", "labels")
    ds_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Model
    cfg = tiny_config(tok.vocab_size)
    cfg.num_labels = ds_tok["train"].features["labels"].num_classes  # 4
    model = BertForSequenceClassification(cfg)

    # Training args
    rank = get_rank()
    tr_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=5e-4,
        ddp_find_unused_parameters=False,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=(rank != 0),
        report_to=[],
        seed=42,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        compute_metrics=compute_metrics,
    )

    # Scientific callbacks
    trainer.add_callback(StepTimer(args.batch, seq_len=128))
    trainer.add_callback(PerRankLogger(args.out, use_tb=True))

    # Train + time
    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()
    if not dist_is_init() or get_rank() == 0:
        print(f"[RANK 0] TRAIN_RUNTIME_SEC={t1 - t0:.3f}", flush=True)

    # ---- EVAL runs on ALL ranks (do NOT guard with rank==0)
    metrics = trainer.evaluate()
    if get_rank() == 0 and "eval_accuracy" in metrics:
        print(f"[RANK 0] EVAL accuracy={metrics['eval_accuracy']:.4f}", flush=True)

    # Optional: synchronize so everyone leaves together cleanly
    if dist_is_init():
        try:
            torch.distributed.barrier()
        except Exception:
            pass

    # Save once
    if not dist_is_init() or get_rank() == 0:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()