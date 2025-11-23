----

## 🧠 Overview

This project documents hands-on experience with:

* **Distributed Systems / HPC**: Working on a **Slurm-based High-Performance Computing (HPC) cluster**.
* **Large Language Models (LLMs)**: Training, fine-tuning, and inference for models like **DistilGPT2, OPT-2.7B, and TinyLLaMA**.
* **Parallelization Techniques**: **Data, model, pipeline, and tensor parallelism** using Python and PyTorch. 

[Image of LLM Parallelization Techniques Diagram]

* **Cluster Utilities**: Scripts for checking resources, launching Slurm jobs, and managing distributed training.

---

## 🛠 Key Features

### Cluster Management
* Automated scripts for evaluating available nodes and partitions (`utils/Recommender.py`)
* Cluster auditing (`utils/checker.py`)
* Secure connection to cluster via SSH (`utils/Connect2Cluster.py`)

### Lab Work
* **Tiny Model Lab**: Lightweight LLM training scripts
* **RAG Lab**: Retrieval-Augmented Generation examples
* **Fine-Tuning Lab**: LoRA & FSDP training on CPU clusters
* **Transfer Learning Lab**: Applied transfer learning on IMDB dataset
* **Simple Model Lab**: Entry-level model training examples

### Documentation
* Step-by-step guides for **Slurm cluster setup** and environment configuration
* Analysis and troubleshooting for multi-node distributed training

---

## ⚡ How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/Mo-Resa77/VT-LLM-Cluster-Training.git](https://github.com/Mo-Resa77/VT-LLM-Cluster-Training.git)
cd VT-LLM-Cluster-Training
2. Set up Python environment
Bash

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
3. Run labs or scripts
Tiny Model Lab:

Bash

python labs/tiny/train_tiny.py
RAG Lab:

Bash

python labs/ragging/rag_example.py
Submit a Slurm job:

Bash

sbatch scripts/slurm_job.sh
Note: Some scripts require a Slurm cluster and may not run locally.

📚 References
PyTorch Distributed Documentation

Hugging Face Transformers

Slurm Workload Manager Documentation

# 🚀 OPT-2.7B FSDP + LoRA Fine-Tuning (CPU Cluster)

This project contains the final scripts and configuration used for the Virginia Tech LLM Cluster Training course, demonstrating distributed fine-tuning of the **facebook/opt-2.7b** Large Language Model on a **CPU-only HPC cluster** environment utilizing **SLURM**.

The core technical achievement is fitting a 5.4 GB model (OPT-2.7B in bfloat16) onto nodes with limited RAM (8 GB/node) using advanced parallelization techniques.

---

## 💡 Core Technique: FSDP + LoRA on CPU

The combination of technologies was specifically chosen to overcome the memory limitations of a CPU-only cluster:

* **PyTorch FSDP (Fully Sharded Data Parallel)**: Shards the model's parameters, gradients, and optimizer states across multiple nodes, ensuring the model fits into the limited memory of each individual node.
* **LoRA (Low-Rank Adaptation)**: Freezes the majority of the pre-trained model weights and injects small, trainable rank-decomposition matrices into the attention layers (`q_proj`, `v_proj`). This dramatically reduces the memory footprint for training.
* **CPU-Only Configuration**: Utilizes the **Gloo backend** for distributed communication, explicitly disabling CUDA, and using `torch.bfloat16` for memory-efficient sharding and computation.

-------------------------------------------------------------------
-------------------------------------------------------------------

## 📂 Final Project Files

| File Name | Description | Key Configuration |
| :--- | :--- | :--- |
| `train_opt27b_fsdp_lora.py` | The main Python script for model loading, tokenization, and training. | FSDP enabled, LoRA targeting OPT attention, CPU-only environment setup, timing logs. |
| `slurm_launch_opt27b.sbatch` | The SLURM job submission script for multi-node distributed execution. | Requests a minimum of **10 nodes** (`#SBATCH -N 10`), uses `srun` for full environment sharing, sets `MASTER_ADDR` for rendezvous. |

---

## 💻 `train_opt27b_fsdp_lora.py` Highlights

### 1. Environment Setup
* Forces CPU-only mode: `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")`
* Sets Gloo backend: `os.environ.setdefault("TORCH_DISTRIBUTED_BACKEND", "gloo")`
* Enables CPU RAM efficiency for loading: `os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "1")`

### 2. Model & LoRA Configuration
* Model loaded with `torch.bfloat16` and `device_map="cpu"`.
* LoRA targets the OPT attention layers: `target_modules=["q_proj", "v_proj"]`.
* Gradient checkpointing is enabled to save memory.

### 3. FSDP Configuration
The `TrainingArguments` specify FSDP settings crucial for multi-node CPU sharding:
* `fsdp="full_shard"`
* `fsdp_config`: Specifies `"fsdp_transformer_layer_cls_to_wrap": "OPTDecoderLayer"` for OPT model structure and `"fsdp_sharding_strategy": 1` (FULL\_SHARD).

### 4. Training
* Uses a small batch size (`per_device_train_batch_size=1`) with a large accumulation steps (`gradient_accumulation_steps=32`) for an effective batch size of 32/world\_size.
* Includes `StepTimerCallback` to monitor and log the duration of each training step.

---

## ⚙️ `slurm_launch_opt27b.sbatch` Details

This script manages the distributed launch across the cluster nodes.

* [cite_start]**Resource Request:** Requires a minimum of 10 nodes (`#SBATCH -N 10`) for the 5.4 GB model to fit within the typical 8 GB RAM per node constraint[cite: 4, 5].
* **Rendezvous:** Calculates `MASTER_ADDR` and `WORLD_SIZE` from SLURM environment variables for `env://` initialization.
* **Execution:** Uses `srun --export=ALL` to launch one task per node (`--ntasks-per-node=1`), ensuring a clean and consistent distributed environment.
* [cite_start]**Data:** Defaults to using a medium open-webtext file: `DATA_FILE="${DATA_FILE:-/data/medium_openwebtext.txt}"`[cite: 3].
* [cite_start]**Logging:** Outputs training details to individual log files per rank: `logs/train_rank${RANK}.log`[cite: 12].

---

## ⚠️ Troubleshooting Note

[cite_start]i says that there is an  issue with the `srun` command i solved it later ,  that needs fixing with external assistance ("still have problem in srun i will fix with eng / metwalli")[cite: 1].

---

## 🎓 Author

**Mohamed Magdy Hagras ,, also huge thanks for eng. metwalli **
*Virginia Tech Summer Training 2025 – VT LLM Cluster Training*
