# Embedding Surgery: A Convex Optimization Framework for Online Ranking Correction

**Embedding Surgery** is a Python framework for **real-time ranking correction in dense retrieval systems**.
It allows minimal, targeted modifications to document embeddings based on **relevance feedback** (from **editorial judgments**, **user clicks**, or **LLM-based pseudo-labels**) without retraining models or re-indexing the corpus.

This repository implements the full experimental pipeline described in the paper

> *Embedding Surgery: A Convex Optimization Framework for Online Ranking Correction*
> (ACM SIGMOD 2026 Submission)

---

## Overview

Dense retrieval systems encode queries and documents into vector embeddings, allowing semantic search via similarity metrics.
However, document embeddings are typically static once indexed, preventing adaptation to new feedback or shifts in user intent.

**Embedding Surgery** bridges this gap by applying lightweight **convex optimization** to adjust document embeddings *at query time*, enforcing ranking constraints derived from feedback while preserving global semantic structure.

---

## Conceptual Summary

Embedding Surgery solves, for each feedback pair ((d_r, d_n)):

[
\min_{\Delta d} \sum_i |\Delta d_i|_2^2
\quad \text{s.t.} \quad
\langle q, d_r + \Delta d_r \rangle \ge \langle q, d_n + \Delta d_n \rangle + \epsilon
]

This convex quadratic program enforces ranking constraints while minimizing distortion in the latent space.

---

## Core Components

| File                         | Description                                                                                                                                                                                                                                                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`generate_embeddings.py`** | Generates dense embeddings for documents and queries using Sentence Transformers and IR datasets (`ir_datasets`). Produces chunked `.npy` files for scalable storage.                                                                                                                         |
| **`index_embeddings.py`**    | Builds a FAISS index from pre-computed embedding chunks and saves internal–external ID mappings.                                                                                                                                                                                              |
| **`embedding_surgery.py`**   | Implements the convex optimization routines for ranking correction.

| **`pipeline.py`**            | Main experimental pipeline integrating retrieval, feedback generation, embedding surgery, and evaluation. Supports three feedback modes: `editorial`, `click_log`, and `llm`.                                                                                                                 |
| **`llm.py`**                 | Implements LLM-based rerankers (BGE and Qwen) for generating pseudo-feedback or weak relevance signals.                                                                                                                                                                                       |
| **`utils.py`**               | Utility functions for FAISS retrieval, TREC-style evaluation (`pytrec_eval`), dataset loading, and mapping between IDs and text.                                                                                                                                                              |

---

## ⚙️ Installation

```bash
git clone https://anonymous.4open.science/r/Embedding-Surgery-88D7
cd Embedding-Surgery-88D7
pip install -r requirements.txt
```

### Main Dependencies

* `torch`, `transformers`
* `sentence-transformers`
* `faiss-gpu`
* `cvxpy`
* `pytrec_eval`
* `ir_datasets`
* `pandas`
* `numpy`

---

## Example Workflow

1. **Generate embeddings** for corpus and queries.
2. **Build a FAISS index** for document embeddings.
3. **Run the pipeline** with one of the feedback approaches.

---

## Generating Embeddings

You can encode **documents** or **queries** from any [ir_datasets](https://ir-datasets.com/) collection or from local TSV files.

```bash
python generate_embeddings.py \
  --model_name "sentence-transformers/msmarco-distilbert-base-tas-b" \
  --dataset_id "msmarco-passage" \
  --batch_size 32 \
  --output_dir "./embeddings" \
  --mode "doc"
```

---

## Building the FAISS Index

```bash
python index_embeddings.py \
  --input_dir ./embeddings/sentence-transformers_msmarco-distilbert-base-tas-b/msmarco-passage \
  --output_dir ./index/
```

---

## Running the Feedback & Surgery Pipeline

Run the full ranking correction loop using one of three feedback sources:

### Feedback Modes

| Mode          | Source                | Description                                                 |
| ------------- | --------------------- | ----------------------------------------------------------- |
| **Editorial** | TREC Qrels            | Ideal feedback derived from human relevance judgments       |
| **Click Log** | Simulated user models | Probabilistic user types: `perfect`, `noisy`, `near_random` |
| **LLM**       | Large Language Models | Uses `Qwen` or `BGE` reranker to infer pairwise constraints |

---

### Example (Click-based Feedback)

```bash
python pipeline.py \
  --approach click_log \
  --surgery_func pos_neg \
  --corpus msmarco-passage \
  --queries msmarco-passage/dev/judged \
  --query_embeddings_path ./embeddings/queries/ \
  --doc_embeddings_path ./embeddings/docs/ \
  --index_path ./index/ \
  --top_k 20 \
  --user_type perfect \
  --eta_bias 0 \
  --n_sim 1000
```

### Example (LLM-based Feedback)

```bash
python pipeline.py \
  --approach llm \
  --reranker qwen \
  --cache_path ./cache/ \
  --surgery_func pos_neg \
  --corpus trec-dl-2020 \
  --queries trec-dl-2020 \
  --query_embeddings_path ./embeddings/queries/ \
  --doc_embeddings_path ./embeddings/docs/ \
  --index_path ./index/ \
  --top_k 20
```

### Example (Editorial “Golden Standard” Feedback)

```bash
python pipeline.py \
  --approach golden_standard \
  --surgery_func pos_neg \
  --corpus trec-dl-2019 \
  --queries trec-dl-2019
```

---

## Evaluation

All retrieval results are evaluated using **pytrec_eval** with:

* `nDCG@5`, `nDCG@10`, `nDCG@20`, `Recall@5`, `Recall@10`, `Reciprocal Rank`

---

## Supported Datasets

* **MS MARCO Passage**
* **TREC Deep Learning (DL 2019, DL 2020, DL-Hard)**
* **TREC CAsT 2019**
* **QSharedRel (MS MARCO Dev)**

---