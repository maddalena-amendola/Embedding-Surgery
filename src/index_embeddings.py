import numpy as np
import faiss
import argparse
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from utils import *

def index_embeddings(
    input_dir: str, 
    output_dir: str
):
    """
    Indexes the embeddings from the input directory into a FAISS HNSW index.
    :param input_dir: Directory containing embedding chunks and doc_id chunks.
    :param output_dir: Directory to save the FAISS index and metadata.
    :param dimension: Dimensionality of the embeddings (default=768).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_ids = get_chunk_ids(input_dir)
    print(f"[indexer] Found {len(chunk_ids)} chunks to process.")

    # Load first chunk to get embedding dimension
    embeddings, _ = load_chunk(input_dir, chunk_ids[0])
    dimension = embeddings.shape[1]
    print(f"[indexer] Embeddings dimension: {dimension}.")

    print("[indexer] Creating FlatIP index...")
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

    all_doc_ids = []               # original string IDs
    all_internal_ids = []         # int64 IDs
    current_id_offset = 0         # track numeric ID assignment

    for chunk_id in tqdm(chunk_ids, desc="Indexing chunks"):
        print(f"[indexer] Processing chunk {chunk_id}...")
        embeddings, doc_ids = load_chunk(input_dir, chunk_id)

        # Create int64 IDs for FAISS
        int_ids = np.arange(current_id_offset, current_id_offset + len(doc_ids), dtype=np.int64)
        index.add_with_ids(embeddings, int_ids)

        all_internal_ids.extend(int_ids)
        all_doc_ids.extend(doc_ids)
        current_id_offset += len(doc_ids)

    # Save the index
    index_file = os.path.join(output_dir, "faiss_flat_ip_index.bin")
    faiss.write_index(index, index_file)

    # Save the mapping: internal_id → original string doc_id
    id_map_file = os.path.join(output_dir, "doc_id_map.npy")
    np.save(id_map_file, np.array(list(zip(all_internal_ids, all_doc_ids)), dtype=object))

    print(f"[indexer] FAISS index saved to {index_file}.")
    print(f"[indexer] ID map saved to {id_map_file}.")
    
    return index

def parse_args():
    parser = argparse.ArgumentParser(
        description="Index embeddings into a FAISS HNSW index."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing embedding chunks and doc_id chunks (required)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the FAISS index and metadata (required)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of embeddings")
    parser.add_argument("--output_dir", required=True, help="Directory to save the FAISS index and metadata (required).")
    
    args = parser.parse_args()  
    print(args.input_dir, args.output_dir)
    _ = index_embeddings(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            )
    
