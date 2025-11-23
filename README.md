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
