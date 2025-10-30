# Embedding Surgery: A Convex Optimization Framework for Online Ranking Correction

**Embedding Surgery** is a Python framework for **real-time ranking correction in dense retrieval systems**.
It allows minimal, targeted modifications to document embeddings based on **relevance feedback** (from **editorial judgments**, **user clicks**, or **LLM-based pseudo-labels**) without retraining models or re-indexing the corpus.

---

## Overview

Dense retrieval systems encode queries and documents into vector embeddings, allowing semantic search via similarity metrics.
However, document embeddings are typically static once indexed, preventing adaptation to new feedback or shifts in user intent.

**Embedding Surgery** bridges this gap by applying lightweight **convex optimization** to adjust document embeddings *at query time*, enforcing ranking constraints derived from feedback while preserving global semantic structure.

Specifically, Embedding Surgery aims to compute a set of updates $\Delta \mathbf{d}\_i$ that satisfy some relevance constraints while minimizing the overall perturbation introduced in the latent space. The optimization problem is defined as:

$\min \quad \sum\_{i=1}^{k} \|\Delta \mathbf{d}\_i\|_2 $
subject to $\quad \langle \mathbf{q}, \mathbf{d}_r + \Delta \mathbf{d}_r \rangle \geq \langle \mathbf{q}, \mathbf{d}\_n + \Delta \mathbf{d}\_n \rangle + \epsilon, \quad \forall (d\_r, d\_n) \in \mathcal{R}$, where:
- $\Delta \mathbf{d}_i$ represents the correction applied to the original embedding $\mathbf{d}_i$;
- $\| \cdot \|_2$ denotes the L2 norm, penalizing large deviations from the original vectors;
- $\epsilon > 0$ defines a margin that enforces separation between the relevance scores of documents $d_r$ and $d_n$.

The objective function encourages minimal movement in the embedding space, ensuring that the adjusted vectors remain close to their original positions, while the constraints guarantee that the resulting ranking is consistent with the feedback. The corrected embeddings are then obtained as:
$\mathbf{d}_i' = \mathbf{d}_i + \Delta \mathbf{d}_i.$

---

## Core Components

| File                         | Description                                                                                                                                                                                                                                                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`generate_embeddings.py`** | Generates dense embeddings for documents and queries using Sentence Transformers and IR datasets (`ir_datasets`). Produces chunked `.npy` files for scalable storage.                                                                                                                         |
| **`index_embeddings.py`**    | Builds a FAISS index from pre-computed embedding chunks and saves internal–external ID mappings.                                                                                                                                                                                              |
| **`embedding_surgery.py`**   | Implements the convex optimization routines for ranking correction.                                                               |
| **`pipeline.py`**            | Main experimental pipeline integrating retrieval, feedback generation, embedding surgery, and evaluation. Supports three feedback modes: `editorial`, `click_log`, and `llm`.                                                                                                                                                                                                                                                                           |                                                                                                            
| **`llm.py`**                 | Implements LLM-based rerankers (BGE and Qwen) for generating pseudo-feedback or weak relevance signals.                           |                                                                                                                                                                             
| **`utils.py`**               | Utility functions for FAISS retrieval, TREC-style evaluation (`pytrec_eval`), dataset loading, and mapping between IDs and text.                                                                                                                                                       

---

## Installation

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
  --model_name "facebook/contriever-msmarco" \
  --dataset_id "msmarco-passage" \
  --output_dir "./embeddings/docs/" \
  --mode "doc"
```

```bash
python generate_embeddings.py \
  --model_name "facebook/contriever-msmarco" \
  --dataset_id "msmarco-passage/trec-dl-2019/judged" \
  --output_dir "./embeddings/queries/" \
  --mode "query" \
  --id_col "query_id"
```

---

## Building the FAISS Index

```bash
python index_embeddings.py \
  --input_dir "./embeddings/facebook_contriever-msmarco/msmarco-passage/" \
  --output_dir "./index/"
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
  --approach "click_log" \
  --surgery_func "symmetric" \
  --corpus "msmarco-passage" \
  --queries "msmarco-passage/trec-dl-2019/judged" \
  --query_embeddings_path "./embeddings/queries/facebook_contriever-msmarco/msmarco-passage_trec-dl-2019_judged/" \
  --doc_embeddings_path "./embeddings/docs/facebook_contriever-msmarco/msmarco-passage/ \
  --index_path "./index/facebook_contriever-msmarco/msmarco-passage/" \
  --top_k 20 \
  --user_type perfect \
  --eta_bias 0 \
  --n_sim 1000
```

### Example (LLM-based Feedback)

```bash
python pipeline.py \
  --approach "llm" \
  --reranker "qwen" \
  --cache_path "./cache/" \
  --surgery_func "symmetric" \
  --corpus "msmarco-passage" \
  --queries "msmarco-passage/trec-dl-2019/judged" \
  --query_embeddings_path "./embeddings/queries/facebook_contriever-msmarco/msmarco-passage_trec-dl-2019_judged/" \
  --doc_embeddings_path "./embeddings/docs/facebook_contriever-msmarco/msmarco-passage/ \
  --index_path "./index/facebook_contriever-msmarco/msmarco-passage/" \
  --top_k 20 \
```

### Example (Editorial Feedback)

```bash
python pipeline.py \
  --approach "editorial" \
  --surgery_func "symmetric" \
  --corpus "msmarco-passage" \
  --queries "msmarco-passage/trec-dl-2019/judged" \
  --query_embeddings_path "./embeddings/queries/facebook_contriever-msmarco/msmarco-passage_trec-dl-2019_judged/" \
  --doc_embeddings_path "./embeddings/docs/facebook_contriever-msmarco/msmarco-passage/ \
  --index_path "./index/facebook_contriever-msmarco/msmarco-passage/" \
  --top_k 20 \
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
