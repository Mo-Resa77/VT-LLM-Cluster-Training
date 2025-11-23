
# using fsdp it divides model layers param etc to fit not just take full copy on each node like ddp so it will work
#but still have problem in my srun 
##---------------------------------------------

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FSDP + LoRA fine-tuning for facebook/opt-2.7b on CPU-only cluster with SLURM.
# - Uses PyTorch FSDP to shard model across nodes (fits 5.4GB model in 8GB RAM/node).
# - LoRA on OPT attention layers (q_proj, v_proj) for efficiency.
# - CPU-only with Gloo backend; env:// init for multi-node.
# - Compatible with Hugging Face Trainer; times phases for reporting.
# - Debug prints for distributed env; saves to ./opt27b_fsdp_lora_cpu.

import os
import socket
import time
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainerCallback

# ---------------- Environment: CPU-only, Gloo, FSDP ----------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # Force CPU-only
os.environ.setdefault("TORCH_DISTRIBUTED_BACKEND", "gloo")
os.environ.setdefault("GLOO_SOCKET_TIMEOUT", "1200")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# FSDP-specific env for CPU efficiency
os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "1")  # Reduce CPU RAM during load

import torch.cuda
torch.cuda.set_device = lambda *_: None  # No-op for CUDA calls

print(f"\n[{socket.gethostname()}] 🛠 ENV DEBUG")
for k in ["TORCH_DISTRIBUTED_BACKEND", "GLOO_SOCKET_TIMEOUT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    print(f"  {k:>24} = {os.environ.get(k)}")
print("-" * 60)

print(f"[{socket.gethostname()}] CUDA available? {torch.cuda.is_available()}")

# ---------------- Init process group (env://) ----------------
if not dist.is_initialized():
    os.environ.setdefault("RANK", os.environ.get("MACHINE_RANK", "0"))
    os.environ.setdefault("WORLD_SIZE", os.environ.get("NUM_MACHINES", "1"))
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "12345"))
    dist.init_process_group(backend="gloo", init_method="env://")
print(f"[dist] rank={dist.get_rank()} world_size={dist.get_world_size()} backend={dist.get_backend()}", flush=True)

rank = dist.get_rank()
world_size = dist.get_world_size()
is_main_process = (rank == 0)

# ---------------- Timing logger ----------------
hostname = socket.gethostname()
rank = int(os.environ.get("RANK", "0"))

def log_timing(phase, duration):
    try:
        log_path = f"logs/timing_rank{rank}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"[{phase}] {duration:.2f} sec\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[{hostname}] ✅ Logged '{phase}' = {duration:.2f} sec -> {log_path}")
    except Exception as e:
        print(f"[{hostname}] ❌ Failed to log timing for '{phase}': {e}")

with open(f"timing_rank{rank}.log", "w") as f:
    f.write(" Timings in sec\n")

# ---------------- Data ----------------
DATA_FILE = os.environ.get("DATA_FILE", "/data/medium_openwebtext.txt")
start = time.time()
dataset = load_dataset("text", data_files={"train": DATA_FILE})["train"]
log_timing("Dataset load", time.time() - start)

# ---------------- Model / Tokenizer (OPT-2.7B + LoRA) ----------------
model_name = "facebook/opt-2.7b"
print(f"[{hostname}] Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"[{hostname}] Loading model: {model_name} (bf16 for memory)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Half memory (5.4GB total)
    low_cpu_mem_usage=True,
    device_map="cpu",  # Explicit CPU
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# Enable gradient checkpointing
try:
    model.gradient_checkpointing_enable()
except Exception:
    pass

# LoRA for OPT: Target q_proj, v_proj (attention layers)
peft_config = LoraConfig(
    r=8,  # Low rank for efficiency
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # OPT attention modules
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# ---------------- Tokenization ----------------
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,  # Shorter for OPT ctx=2048, but CPU-friendly
    )

start = time.time()
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
log_timing("Tokenization", time.time() - start)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------- Trainer (FSDP + DDP) ----------------
class StepTimerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        dt = time.time() - getattr(self, "_t0", time.time())
        print(f"[{socket.gethostname()}] Step {state.global_step} took {dt:.3f} sec")

output_dir = "./opt27b_fsdp_lora_cpu"

# FSDP config for CPU/multi-node sharding
fsdp_config = {
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",  # Auto-wrap OPT layers
    "fsdp_backward_prefetch": "BACKWARD_PRE",
    "fsdp_forward_prefetch": True,
    "fsdp_offload_params": True,  # Offload to CPU if needed
    "fsdp_sharding_strategy": 1,  # FULL_SHARD
    "fsdp_state_dict_type": "FULL_STATE_DICT",
    "fsdp_transformer_layer_cls_to_wrap": "OPTDecoderLayer",  # OPT-specific
}

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,  # Small for CPU
    gradient_accumulation_steps=32,  # Effective batch=32/world_size
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=1,
    logging_steps=50,
    learning_rate=5e-5,
    no_cuda=True,  # CPU-only
    bf16=True,  # bfloat16 for OPT
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,  # CPU-friendly
    fsdp="full_shard",  # Enable FSDP
    fsdp_config=fsdp_config,
    ddp_backend="gloo",  # CPU comm
    ddp_find_unused_parameters=False,
)

start = time.time()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[StepTimerCallback()],
)
log_timing("Trainer setup", time.time() - start)

# ---------------- Train ----------------
start = time.time()
trainer.train()
log_timing("Training", time.time() - start)

# Sync before save
if dist.is_initialized():
    dist.barrier()

# ---------------- Save (main only) ----------------
start = time.time()
if trainer.is_world_process_zero():
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model + LoRA adapter saved to {output_dir}")
log_timing("Model Save", time.time() - start)

# Teardown
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()