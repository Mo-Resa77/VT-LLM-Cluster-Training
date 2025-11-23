#!/usr/bin/env python3
"""
Ultra-light CPU RAG (Retrieval-Augmented Generation) — Teaching Demo
====================================================================

AIM (why this file exists)
--------------------------
This script demonstrates the core mechanics of RAG on a CPU-only cluster:

1) "Train" the retriever by fitting a TF–IDF vectorizer over a small text corpus.
   • This is the training phase for retrieval (not the generator (LLM)).
   • It learns a vocabulary and IDF weights to score relevance.

2) Retrieve top-k passages for a user query.
   • Primary retriever: TF–IDF + cosine similarity (fast and deterministic).
   • Fallback retriever: token-overlap (when scikit-learn isn't available).

3) Generate an answer conditioned on retrieved context with FLAN-T5-Small.
   • We prompt a small, pre-trained sequence-to-sequence model.
   • No fine-tuning of the generator is performed here.

4) (Optional) Parallelize inference across multiple processes with torchrun.
   • Queries can be sharded across ranks via LOCAL_RANK / WORLD_SIZE.
   • This illustrates CPU data-parallel inference on SLURM.

Design choices to keep the demo stable and fast on CPU:
• Small corpus slice (AG News) and a small generator (FLAN-T5-Small).
• Simple, deterministic prompt template to reduce output variance.
• Clear knobs for speed/quality trade-offs: --subset, --k, --max_new_tokens.

-------------------------------------------------------------------------------
USAGE (single query):
  export HF_HOME=$HOME/.cache/hf_rag
  python labs/ragging/rag_example.py --dry_run --query "What is deep learning?"

USAGE (batched + torchrun on one node):
  torchrun --nproc_per_node=2 labs/ragging/rag_example.py \
    --dry_run --queries_file labs/ragging/queries.txt --k 3 --max_new_tokens 64

ENV NOTES:
  HF_HOME      -> base cache dir for tokenizer/model/dataset (recommended).
  LOCAL_RANK   -> set by torchrun; this worker's index (0..WORLD_SIZE-1).
  WORLD_SIZE   -> set by torchrun; total worker processes.
  MASTER_*     -> (optional on single-node) torchrun rendezvous settings.

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
import re
import time
import math
import argparse
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------------------------------------------------------
# 1) Normalization utilities
# -----------------------------------------------------------------------------
# Compile a simple regex for "one or more whitespace" once at import time.
_WS_RE = re.compile(r"\s+")


def norm(txt: str) -> str:
    """
    Normalize input text by lowercasing and collapsing whitespace.

    Why: Normalization makes retrieval more robust and stable (e.g., "USA" vs "usa") - TF-IDF,
    and it helps both TF–IDF and the fallback overlap scorer behave consistently.
    """
    return _WS_RE.sub(" ", txt.lower()).strip()


def simple_overlap_score(q: str, doc: str) -> float:
    """
    Fallback similarity measure based on token overlap.

    We compare unique tokens (sets) and scale by sqrt(|Q| * |D|) to avoid favoring
    very long texts. This is intentionally simple and dependency-free.

    Returns:
        A score in [0, 1]-ish (not strictly bounded) — larger is "more similar".
    """
    q_tokens = set(norm(q).split())
    d_tokens = set(norm(doc).split())
    if not q_tokens or not d_tokens:
        return 0.0
    inter = len(q_tokens & d_tokens)
    return inter / math.sqrt(len(q_tokens) * len(d_tokens))


# -----------------------------------------------------------------------------
# 2) Retriever (TF–IDF or fallback)
# -----------------------------------------------------------------------------
class Retriever:
    """
    Encapsulates the retrieval step.

    On initialization:
      • Tries to fit a TF–IDF vectorizer over the corpus ("training" the retriever).
      • If scikit-learn isn't available, falls back to a token-overlap scorer.

    Attributes:
        docs:  List[str]        The corpus of passages.
        kind:  str              "tfidf" or "fallback"
        vec:   TfidfVectorizer  Only when TF–IDF is available.
        mat:   sparse matrix    Document-term matrix (TF–IDF); shape: [N_docs, Vocab]
    """

    def __init__(self, docs: List[str]) -> None:
        self.docs: List[str] = docs
        self.kind: str = "fallback"
        self.vec = None
        self.mat = None

        try:
            # Import inside the try so the file still runs if sklearn isn't installed.
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Limit max_features for speed/memory; include bigrams for extra signal.
            self.vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))

            # This is the "training" step for retrieval: learn vocab + IDF weights.
            # It returns a sparse document-term matrix.
            self.mat = self.vec.fit_transform(docs)
            self.kind = "tfidf"
        except Exception as e:
            # Cleanly degrade to the simple overlap scorer.
            self.vec = None
            self.mat = None
            print(
                f"[WARN] scikit-learn unavailable, using overlap fallback: {e}",
                file=sys.stderr,
            )

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """
        Return the top-k (doc_index, score) for a query.

        TF–IDF path:
            Transform the query into the same TF–IDF space and compute similarity
            with all documents. We use (mat @ qv) which approximates cosine if
            rows are L2-normalized (the default in TfidfVectorizer).

        Fallback path:
            Score each doc with the overlap heuristic and sort.

        Args:
            query: The question string.
            k    : The number of passages to return.

        Returns:
            A list of (document_index, score), sorted by descending score.
        """
        if self.kind == "tfidf":
            import numpy as np  # Local import to keep base import light

            qv = self.vec.transform([query])  # shape: [1, Vocab]
            # Sparse dot product: [N_docs, Vocab] @ [Vocab, 1] -> [N_docs]
            sims = (self.mat @ qv.T).toarray().ravel()
            topk_idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in topk_idx]

        # Fallback: O(N_docs) scan with simple token overlap
        scored = [(i, simple_overlap_score(query, d)) for i, d in enumerate(self.docs)]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# -----------------------------------------------------------------------------
# 3) Prompt builder
# -----------------------------------------------------------------------------
def build_prompt(query: str, passages: List[str]) -> str:
    """
    Construct a compact, deterministic instruction for the generator.

    Why deterministic? On CPU, shorter, stable prompts reduce latency and
    variability between runs, which is ideal for teaching and benchmarking.
    """
    context_block = "\n\n".join(f"- {p}" for p in passages) # String joined that includes all the context of passages
    return (
        "Answer the question concisely using the context.\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


# -----------------------------------------------------------------------------
# 4) Main (argument parsing, data loading, retrieval, generation)
# -----------------------------------------------------------------------------
def main() -> None:
    # ---- 4.0 CLI arguments ---------------------------------------------------
    ap = argparse.ArgumentParser(
        description="CPU-only RAG demo: TF–IDF retriever + FLAN-T5-Small generator."
    )
    ap.add_argument("--subset", type=int, default=2000, help="Corpus size (train[:N]).")
    ap.add_argument("--k", type=int, default=3, help="Top-k passages to retrieve.")
    ap.add_argument("--query", type=str, default=None, help="Single query string.")
    ap.add_argument(
        "--queries_file", type=str, default=None, help="Path to a file of queries (one per line)."
    )
    ap.add_argument(
        "--batch", type=int, default=4, help="Batch size for file mode (looped, not vectorized)."
    )
    ap.add_argument(
        "--max_new_tokens", type=int, default=64, help="Max tokens to generate per answer."
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a tiny slice (train[:2000]) for fast demonstration.",
    )
    args = ap.parse_args()

    # ---- 4.1 Assign cache directory (shared across steps) --------------------
    # Prefer HF_HOME (recommended on clusters); otherwise, use a local folder.
    cache_root = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".hf_rag_cache"))

    # ---- 4.2 Load a small corpus (fast, cached) ------------------------------
    # For a quick demo, cap to 2k examples when --dry_run is set.
    split = f"train[:{min(args.subset, 2000)}]" if args.dry_run else f"train[:{args.subset}]"
    ds = load_dataset("ag_news", split=split, cache_dir=os.path.join(cache_root, "ds"))

    # Build our in-memory corpus as simple strings (title + text where available).
    corpus: List[str] = [
        f"{row.get('title','')} - {row['text']}".strip(" -") for row in ds  # type: ignore[index]
    ]

    # ---- 4.3 Train the retriever (fit TF–IDF) --------------------------------
    t0 = time.time()
    retr = Retriever(corpus)
    train_time = time.time() - t0
    print(
        f"[retriever] kind={retr.kind} "
        f"trained_on={len(corpus)} docs in {train_time:.2f}s"
    )

    # ---- 4.4 Load the generator (small T5) -----------------------------------
    # Tokenizer converts text -> model inputs; model generates text.
    tok = AutoTokenizer.from_pretrained(
        "google/flan-t5-small", cache_dir=os.path.join(cache_root, "tok")
    )
    gen = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small", cache_dir=os.path.join(cache_root, "gen")
    )
    gen.eval()  # Disable gradients for faster inference and lower memory.

    # ---- 4.5 Single-query answering function --------------------------------
    def answer_one(q: str) -> str:
        """
        Retrieve top-k contexts for q, build a prompt, and generate an answer.
        This is the core RAG flow for a single question.
        """
        hits = retr.search(q, k=args.k)
        contexts = [corpus[i] for i, _ in hits]

        prompt = build_prompt(q, contexts)
        # Tokenize to PyTorch tensors; model expects input_ids/attention_mask keys.
        inputs = tok(prompt, return_tensors="pt")

        with torch.no_grad():
            out = gen.generate(**inputs, max_new_tokens=args.max_new_tokens)

        # Convert token IDs back to readable text and strip special tokens.
        return tok.decode(out[0], skip_special_tokens=True)

    # ---- 4.6 Single-query path ----------------------------------------------
    if args.query:
        ans = answer_one(args.query)
        print("\nQ:", args.query)
        print("A:", ans)
        sys.exit(0)

    # ---- 4.7 Batched path (supports torchrun sharding) -----------------------
    if args.queries_file and os.path.exists(args.queries_file):
        # Read queries from a file (one per line, empty lines ignored).
        with open(args.queries_file, "r", encoding="utf-8") as fh:
            queries: List[str] = [line.strip() for line in fh if line.strip()]

        # If launched via torchrun, these env vars are set automatically:
        #   WORLD_SIZE -> total number of worker processes
        #   LOCAL_RANK -> index of this worker (0..WORLD_SIZE-1)
        rank = int(os.environ.get("LOCAL_RANK", 0)) # Playaround with Local Rank
        world = int(os.environ.get("SLURM_NTASKS", 1)) # Tested

        # Strided sharding: worker 0 takes lines 0, world, 2*world, ...
        shard = queries[rank::world]
        print(f"[rank {rank}/{world}] processing {len(shard)} queries")

        # Process in small Python batches for readability (not vectorized).
        for i in range(0, len(shard), args.batch):
            for q in shard[i : i + args.batch]:
                ans = answer_one(q)
                print(f"\nQ: {q}\nA: {ans}")
        sys.exit(0)

    # ---- 4.8 No work specified ----------------------------------------------
    print("Nothing to do: provide --query '...' or --queries_file <path>.")
    return


# -----------------------------------------------------------------------------
# Script entry point (only runs when executed as a program)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()


'''
Interp:
Your corpus is AG News headlines/articles, not a textbook or FAQ about ML. So when you ask a generic definitional question, TF‑IDF hunts for literal word overlap in news. It finds “deep” in deep‑sea, “learning” in lifelong learning, etc.

TF‑IDF is purely lexical. It doesn’t know “deep learning” (ML) ≠ “deep sea”. With few or no exact matches, the top “similar” docs can be tangential.

Your prompt forces grounding in the retrieved context. You tell FLAN‑T5 to “Answer … using the context.” If the context is an unrelated news snippet, the model obediently answers about that snippet—hence the nautical response.

Small model + tiny slice. flan‑t5‑small on CPU and train[:2000] means limited knowledge and limited recall. You didn’t fine‑tune the generator, so it won’t override bad context with encyclopedic memory.
'''