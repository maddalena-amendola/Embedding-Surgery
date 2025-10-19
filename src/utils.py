import numpy as np
import pytrec_eval
from collections import defaultdict, Counter
import faiss
import gzip
import json
import os
import ir_datasets
from pathlib import Path
import ir_datasets
import itertools
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random

random.seed(42)

def read_json(filename):    
    
    with gzip.open(filename, "rt") as f:
        obj = json.load(f)
    
    return obj

def write_json(obj, filename):    
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)
    return


def run_faiss_retrieval(index, doc_ids, query_embeddings, query_ids, top_k=10):
    
    #print("[FAISS] Running search...")
    scores, indices = index.search(query_embeddings, top_k)

    #print("[FAISS] Building run dictionary...")
    run = {}
    for i, qid in enumerate(query_ids):
        qid = str(qid)
        run[qid] = {}
        for doc_idx, score in zip(indices[i], scores[i]):
            doc_id = str(doc_ids[doc_idx])
            run[qid][doc_id] = float(score)
            
    return run

def evaluate(qrels, run):
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_5', 'ndcg_cut_10', 'recall_5', 'recall_10', 'ndcg_cut_15', 'ndcg_cut_20', 'recall_15', 'recall_20', 'recip_rank'})
    results = evaluator.evaluate(run)    
    
    return results

def bubble_sort(arr, docids):
    pairs_indexes, pairs_ids = [], []
    # Outer loop to iterate through the list n times
    for n in range(len(arr) - 1, 0, -1):
        
        # Initialize swapped to track if any swaps occur
        swapped = False  

        # Inner loop to compare adjacent elements
        for i in range(n):
            if arr[i] > arr[i + 1]:
                pairs_indexes.append((i, i+1))
                pairs_ids.append((docids[i], docids[i+1]))
                # Swap elements if they are in the wrong order
                arr[i], arr[i + 1] = arr[i + 1], arr[i]                
                docids[i], docids[i + 1] = docids[i + 1], docids[i]
                # Mark that a swap has occurred
                swapped = True
        
        # If no swaps occurred, the list is already sorted
        if not swapped:
            break
        
    return pairs_indexes, pairs_ids

def build_lookup_table(dataset_id: str, id_col: str = 'query_id', txt_col: str = 'text', is_query: bool = False, use_title: bool = False):
    """
    Build a lookup table (id -> text) for queries or documents.
    
    :param dataset_id: The dataset ID or path to the local TSV file.
    :param id_col: The column name for the ID (either 'query_id' or 'doc_id').
    :param txt_col: The column name for the text content ('text' by default).
    :param is_query: If True, the dataset is assumed to be for queries; otherwise, for documents.
    :param use_title: Whether to prepend the title to the text (for documents).
    :return: A dictionary where the key is the id and the value is the text.
    """
    lookup_table = {}

    if Path(dataset_id).is_file():
        # Handle the TSV file source
        with open(dataset_id, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                doc_id = row.get(id_col)
                text = row.get(txt_col, "")
                if use_title and "title" in row and row["title"]:
                    text = row["title"] + " " + text
                lookup_table[doc_id] = text

    else:
        # Handle the IR dataset source via ir_datasets
        if is_query:
            ds = ir_datasets.load(dataset_id).queries_iter()
        else:
            ds = ir_datasets.load(dataset_id).docs_iter()

        for item in ds:
            text = getattr(item, txt_col)
            if use_title and hasattr(item, "title") and item.title:
                text = item.title + " " + text
            lookup_table[getattr(item, id_col)] = text

    return lookup_table

def load_chunk(input_dir, chunk_id):
    """
    Loads a single embedding chunk and its corresponding doc_id chunk.
    :param input_dir: Directory containing embedding and doc_id chunks.
    :param chunk_id: Chunk ID to load.
    :return: (embeddings, doc_ids)
    """
    emb_file = os.path.join(input_dir, f"embeddings_chunk_{chunk_id}.npy")
    doc_id_file = os.path.join(input_dir, f"doc_ids_chunk_{chunk_id}.npy")

    embeddings = np.load(emb_file)
    doc_ids = np.load(doc_id_file, allow_pickle=True)

    return embeddings, doc_ids

def get_chunk_ids(input_dir):
    """
    Retrieves all chunk IDs from the input directory.
    :param input_dir: Directory containing embedding and doc_id chunks.
    :return: List of chunk IDs.
    """
    chunk_ids = []
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.startswith("embeddings_chunk_") and file_name.endswith(".npy"):
            chunk_id = file_name.split("_")[-1].split(".")[0]
            chunk_ids.append(chunk_id)
    return chunk_ids

def load_embeddings_chunk(embeddings_path, chunk_id):
    """
    Helper function to load a chunk of embeddings and corresponding doc_ids.
    """
    embeddings, ids = load_chunk(embeddings_path, chunk_id)  # shape: (chunk_size, d)
    return embeddings, ids

def load_all_embeddings(embeddings_path, n_workers=4):
    """
    Loads all embeddings and their corresponding IDs from a given path, potentially in parallel to speed up the process.
    
    :param embeddings_path: Path to the embeddings files.
    :param n_workers: Number of parallel workers for loading the embeddings.
    :return: A tuple (all_embeddings, all_ids), where:
             - all_embeddings: A concatenated numpy array of all embeddings.
             - all_ids: A concatenated numpy array of all document IDs.
    """
    chunk_ids = get_chunk_ids(embeddings_path)
    
    # Use ThreadPoolExecutor for parallel processing of chunks
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Use tqdm for progress tracking
        all_embeddings, all_ids = zip(*tqdm(executor.map(lambda chunk_id: load_embeddings_chunk(embeddings_path, chunk_id), chunk_ids), total=len(chunk_ids)))
    
    # Concatenate all embeddings and ids at once
    all_embeddings = np.vstack(all_embeddings)
    all_ids = np.hstack(all_ids)
    
    return all_embeddings, all_ids

def get_msmarco_shared_queries():
    
    ds = ir_datasets.load('msmarco-passage/dev/judged')
    
    qrels = defaultdict(set)
    for q in ds.qrels_iter():
        if q.relevance == 1:
            qrels[q.doc_id].add(q.query_id)
            
    repeated_docs = {k:sorted(list(v), reverse=False) for k,v in qrels.items() if len(v)>1}
    
    counter = Counter(itertools.chain.from_iterable(repeated_docs.values()))
    queries_to_consider = set([k for k,v in counter.items() if v>1])
    map_queries = dict()
    for k, v in repeated_docs.items():
        if len(set.intersection(queries_to_consider, set(v))) == 0:
            queries_to_consider.add(v[0])
        selected = list(set.intersection(queries_to_consider, set(v)))[0]
        map_queries[selected] = set(v) - {selected}
            
    return list(set(itertools.chain.from_iterable(repeated_docs.values()))), sorted(list(queries_to_consider), reverse=False), map_queries