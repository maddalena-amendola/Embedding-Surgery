import os
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from sentence_transformers import SentenceTransformer
import ir_datasets
import argparse
import csv
from pathlib import Path


class IRIterableDataset(IterableDataset):
    """
    An IterableDataset that reads documents from either an IR dataset (via ir_datasets)
    or a local TSV file in a streaming fashion.
    
    Each item is a tuple: (doc_id, text), where text may include a title if requested.
    """
    def __init__(self, dataset_id: str, id_col: str = 'query_id', txt_col: str = 'text', use_title: bool = False):
        super().__init__()
        self.use_title = use_title
        self.dataset_id = dataset_id
        self.id_col = id_col
        self.txt_col = txt_col

        if Path(dataset_id).is_file():
            self.source_type = "tsv"
        else:
            self.source_type = "ir_dataset"
            self.ds = ir_datasets.load(dataset_id).docs_iter()

    def __iter__(self):
        if self.source_type == "ir_dataset":
            for doc in self.ds:
                text = getattr(doc, self.txt_col)
                if self.use_title and hasattr(doc, "title") and doc.title:
                    text = doc.title + " " + text
                yield (getattr(doc, self.id_col), text)

        elif self.source_type == "tsv":
            with open(self.dataset_id, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    doc_id = row.get(self.id_col)
                    text = row.get(self.txt_col, "")
                    if self.use_title and "title" in row and row["title"]:
                        text = row["title"] + " " + text
                    yield (doc_id, text)

class IRQueryIterableDataset(IterableDataset):
    """
    An IterableDataset that reads queries from either an IR dataset (via ir_datasets)
    or a local TSV file in a streaming fashion.
    
    Each item is a tuple: (query_id, text).
    """
    
    def __init__(self, dataset_id: str, id_col: str = 'query_id', txt_col: str = 'text'):
        super().__init__()
        self.id_col = id_col
        self.txt_col = txt_col
        self.dataset_id = dataset_id
        
        if Path(dataset_id).is_file():
            self.source_type = "tsv"
        else:
            self.source_type = "ir_dataset"
            self.ds = ir_datasets.load(dataset_id).queries_iter()

    def __iter__(self):
        if self.source_type == "ir_dataset":
            for query in self.ds:
                yield (getattr(query, self.id_col), getattr(query, self.txt_col))

        elif self.source_type == "tsv":
            with open(self.dataset_id, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    query_id = row.get(self.id_col)
                    text = row.get(self.txt_col, "")
                    yield (query_id, text)


def collate_fn(batch):
    """
    batch: list of (doc_id, text) pairs
    Returns:
      doc_ids: list of doc_ids (strings)
      texts: list of texts (strings)
    """
    doc_ids = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return doc_ids, texts

def main(
    model_name: str,
    dataset_id: str,
    batch_size: int = 16,
    flush_size: int = 1000000,
    output_dir: str = "ir_embeddings_chunks",
    use_title: bool = False,
    normalize_embeddings=False,
    id_col: str = 'doc_id',
    txt_col: str = 'text',
    doc_prompt= None,
    query_prompt= None,
    mode: str = 'doc'
):
    """
    :param model_name: which Sentence Transformers model to use (required).
    :param dataset_id: which IR dataset to load from ir_datasets (required).
    :param batch_size: how many samples (docs) per cuda per forward pass (default=16).
    :param flush_size: flush embeddings/doc_ids to disk after this many total docs (default=1000).
    :param output_dir: final sub-directory to store chunked embeddings (default='ir_embeddings_chunks').
    :param use_title: if True, prepend the document title (if present) to the text.
    """

    # --------------------------------------------------------------------------
    # Construct final output path: output/<safe_model_name>/<safe_dataset_id>/<output_dir>
    # Replace slashes in model_name or dataset_id with underscores to avoid nested dirs.
    # --------------------------------------------------------------------------
    safe_model_name = model_name.replace("/", "_")
    
    if Path(dataset_id).is_file(): 
        safe_dataset_id = dataset_id.split('/')[-1].split('.')[0]
    else:
        safe_dataset_id = dataset_id.replace("/", "_")

    final_output_dir = os.path.join(output_dir, safe_model_name, safe_dataset_id)
    os.makedirs(final_output_dir, exist_ok=True)

    # 1) Build an IterableDataset for IR docs or queries based on mode
    if mode == 'doc':
        dataset = IRIterableDataset(dataset_id=dataset_id, id_col=id_col, txt_col=txt_col, use_title=use_title)
    elif mode == 'query':
        dataset = IRQueryIterableDataset(dataset_id=dataset_id, id_col=id_col, txt_col=txt_col)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'doc' or 'query'.")

    # 2) Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0  # can set >0 if you want parallel reading
    )

    # 3) Load and prepare Sentence Transformers model
    model = SentenceTransformer(model_name)
    model.eval().cuda()

    # Start the multi-process pool for inference
    pool = model.start_multi_process_pool()

    # Buffers on main process
    doc_ids_buffer = []
    embed_buffer = []
    total_docs_in_buffer = 0
    chunk_id = 0

    # Inference loop
    with torch.no_grad():
        for doc_ids, texts in dataloader:
            # Use encode_document for docs and encode_query for queries
            if mode == 'doc':
                embeddings = model.encode_document(
                    texts,
                    show_progress_bar=True,
                    normalize_embeddings=normalize_embeddings,
                    batch_size=batch_size,
                    device=['cuda', 'cuda', 'cuda', 'cuda'],
                    pool=pool,
                    chunk_size=512,
                    doc_prompt=doc_prompt
                )
            elif mode == 'query':
                embeddings = model.encode_query(
                    texts,
                    show_progress_bar=True,
                    normalize_embeddings=normalize_embeddings,
                    batch_size=batch_size,
                    device=['cuda', 'cuda', 'cuda', 'cuda'],
                    pool=pool,
                    chunk_size=512,
                    query_prompt=query_prompt
                )
            
            # Example of collecting embeddings and doc_ids
            doc_ids_buffer.append(doc_ids)
            embed_buffer.append(embeddings)
            total_docs_in_buffer += len(doc_ids)

            # Flush if we exceed flush_size
            if total_docs_in_buffer >= flush_size:
                # Concatenate
                all_doc_ids = sum(doc_ids_buffer, [])
                all_embeds = np.concatenate(embed_buffer, axis=0)

                # Save doc_ids and embeddings
                doc_id_file = os.path.join(final_output_dir, f"doc_ids_chunk_{chunk_id}.npy")
                emb_file = os.path.join(final_output_dir, f"embeddings_chunk_{chunk_id}.npy")

                np.save(doc_id_file, np.array(all_doc_ids, dtype=object))
                np.save(emb_file, all_embeds)
                print(f"[main process] Flushed {len(all_doc_ids)} docs to {doc_id_file}, {emb_file}")

                # Reset
                doc_ids_buffer.clear()
                embed_buffer.clear()
                total_docs_in_buffer = 0
                chunk_id += 1

    # 5) Final flush for leftover data
    all_doc_ids = sum(doc_ids_buffer, [])
    all_embeds = np.concatenate(embed_buffer, axis=0)

    doc_id_file = os.path.join(final_output_dir, f"doc_ids_chunk_{chunk_id}.npy")
    emb_file = os.path.join(final_output_dir, f"embeddings_chunk_{chunk_id}.npy")

    np.save(doc_id_file, np.array(all_doc_ids, dtype=object))
    np.save(emb_file, all_embeds)
    print(f"[main process] Final flush of {len(all_doc_ids)} docs to {doc_id_file}, {emb_file}")
    
    model.stop_multi_process_pool(pool)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode IR dataset docs with a Sentence Transformers model, "
                    "distributing across multiple cudas. Optionally use doc titles."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name/path for Sentence Transformers (required)."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="IR dataset ID (e.g. msmarco-passage/train) for ir_datasets (required)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per device (default: 16)."
    )
    parser.add_argument(
        "--flush_size",
        type=int,
        default=1000000,
        help="Flush to disk after encoding this many docs (default: 1000)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ir_embeddings_chunks",
        help="Final subdirectory name for chunked embeddings (default: ir_embeddings_chunks)."
    )
    parser.add_argument(
        "--use_title",
        action="store_true",
        help="If specified, prepend doc.title to doc.text (if available)."
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="If specified, normalize the embeddings."
    )
    parser.add_argument(
        "--doc_prompt",
        type=str,
        default=None,
        help="Path to file containing doc prompt."
    )    
    parser.add_argument(
        "--query_prompt",
        type=str,
        default=None,
        help="Path to file containing query prompt."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='doc',
        help="Equal to 'doc' to create embeddings for docs, 'query' for queries."
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default='doc_id',
        help='Name of the column for the corpus or queries IDs'
    )
    parser.add_argument(
        "--txt_col",
        type=str,
        default='text',
        help='Name of the column for the corpus or queries text'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        model_name=args.model_name,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        flush_size=args.flush_size,
        output_dir=args.output_dir,
        use_title=args.use_title,
        normalize_embeddings=args.normalize_embeddings,
        doc_prompt=args.doc_prompt,
        query_prompt=args.query_prompt,
        mode=args.mode,
        id_col=args.id_col,
        txt_col=args.txt_col
    )

