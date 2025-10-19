from utils import *
from embedding_surgery import *
import numpy as np
import faiss
import argparse
import random
from llm import QwenReranker, BGEReranker
import ir_datasets
import time
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

def click_log(qrels_df, click_distributions, run_qid, top_k, user_type, eta_bias):
    
    # build the run dataframe
    rows = []
    for query_id, docs in run_qid.items():
        for doc_id, score in docs.items():
            rows.append({'query_id': query_id, 'doc_id': doc_id, 'score': score})
    run_df = pd.DataFrame(rows)
    run_df.score = run_df.score.astype(float)
    run_df["rank"] = run_df.groupby("query_id")["score"].rank(ascending=False)
    
    # join the run and qrels info
    cut_run = run_df.loc[run_df["rank"] <= top_k]
    cut_run = cut_run.merge(qrels_df[["query_id", "doc_id", "relevance"]], how="left")
    cut_run = cut_run.fillna(0)
    cut_run = cut_run[["query_id", "doc_id", "rank", "relevance"]]
    
    # compute biases
    cut_run["rclick"] = cut_run["relevance"].map(click_distributions[user_type])
    cut_run["bias"] = (1 / cut_run["rank"]) ** eta_bias
    cut_run["pclick"] = cut_run["rclick"] * cut_run["bias"]
    
    # simulate the click
    srun = cut_run.copy()
    srun["click"] = np.random.random(size=len(cut_run))
    srun["click"] = srun["click"] < srun["pclick"]
    
    # take the first relevant click
    true_click = next(iter(srun[srun['click']].index), -1)
    
    if true_click > 0:
        not_relevant_doc = srun.doc_id.values[true_click-1]
        relevant_doc = srun.doc_id.values[true_click]
        return [(not_relevant_doc, relevant_doc)] 
    else:
        return []
    
def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--reranker", required=False, help="LLM-based reranker to use.")  
    parser.add_argument("--query_embeddings_path", required=True, help="Directory containing embedding chunks and doc_id chunks for queries (required)")
    parser.add_argument("--index_path", required=True, help="Directory to save the FAISS index and metadata (required).")
    parser.add_argument("--doc_embeddings_path", required=True, help="Directory containing embedding chunks and doc_id chunks for documents (required).")
    parser.add_argument("--top_k", required=True, default=20, help="Number of document to (required).", type=int)
    parser.add_argument("--surgery_func", required=True, help="Function to modify relevant, not relevant or both documents (required).")
    parser.add_argument("--corpus", required=True, help="Dataset name for ir_dataset library (required).")
    parser.add_argument("--queries", required=True, help="Query set name for ir_dataset library or path to tsv file (required).")
    parser.add_argument("--model", required=True, help="Model name/path for Sentence Transformers (required).")
    parser.add_argument("--approach", required=True, help="Feedback approach: golden standard or llm")
    parser.add_argument("--user_type", required=False, help="User type to create log")
    parser.add_argument("--eta_bias", required=False, default=0, help="Bias parameter", type=int)    
    parser.add_argument("--n_sim", required=False, default=1000, help="Number of user feedback simulations", type=int)
    parser.add_argument("--cache_path", required=False, help="Path for LLM models", type=int)
    
    return parser.parse_args() 

def main(args, results_dir):
    
    random.seed(42)  
    
    approach = args.approach.lower()

    print("Loading index...")
    index = faiss.read_index(args.index_path + "faiss_flat_ip_index.bin")   
        
    print("Loading document embeddings...")
    doc_embeddings, _ = load_all_embeddings(args.doc_embeddings_path)
    doc_ids_map = np.load(args.index_path+'doc_id_map.npy', allow_pickle=True)
    doc_ids = doc_ids_map[:,0]
        
    print("Loading query embeddings...")
    query_embeddings, query_ids = load_all_embeddings(embeddings_path=args.query_embeddings_path)
    query_ids_map = dict(zip(query_ids, range(len(query_ids))))

    print("Preparing queries...")
    if 'trec-cast_v1_2019_judged.tsv' in args.queries:
        dataset = ir_datasets.load('trec-cast/v1/2019/judged')
    else:
        dataset = ir_datasets.load(args.queries)
        
    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[str(qrel.query_id)][str(qrel.doc_id)] = int(qrel.relevance)
    
    if 'cast' in args.queries.lower():
        topics = list(sorted(list(set([query.topic_number for query in dataset.queries_iter()])), reverse=False))
        queries = [str(elem)+'_1' for elem in topics]
    elif 'msmarco-passage/dev/judged' in args.queries:
        all_queries, queries, _ = get_msmarco_shared_queries() #np.load('msmarco-passage-shared.npy', allow_pickle=True)
        indices = [query_ids_map.get(elem) for elem in all_queries]
        
        query_embeddings = query_embeddings[indices]
        query_ids = query_ids[indices]
        query_ids_map = dict(zip(query_ids, range(len(query_ids))))
        qrels = {k:v for k,v in qrels.items() if k in query_ids}
    else:
        queries = list(query_ids.copy())
        
    if approach == 'click_log':
        queries = list(random.choices(queries, k=args.n_sim))
    
    # Setup
    if approach == 'llm':       
        print('Loading lookup tables...') 
        
        if 'cast' in args.corpus:
            id_col = 'id'
        else:
            id_col = 'doc_id'
        doc_lookup = build_lookup_table(args.corpus, id_col=id_col, txt_col='text')
        
        if 'cast' in args.queries.lower():
            txt_col = 'manual_rewritten_utterance'
        else:
            txt_col = 'text'
            
        query_lookup = build_lookup_table(args.queries, id_col='query_id', txt_col=txt_col, is_query=True) 
        
        print('Loading LLM-based reranker...')
        # Initialize reranker
        reranker_class = {
            'qwen': QwenReranker(cache_path),
            'bge': BGEReranker(cache_path)
        }
        reranker = reranker_class.get(args.reranker.lower())
    
    elif approach == 'click_log':        
        docid_index = dict(zip(doc_ids_map[:, 1], doc_ids_map[:, 0]))
        
        rows = []
        for query_id, docs in qrels.items():
            for doc_id, relevance in docs.items():
                rows.append({'query_id': query_id, 'doc_id': str(docid_index.get(doc_id)), 'relevance': relevance})
        qrels_df = pd.DataFrame(rows)
        
        click_distributions = {
            "perfect": {0: 0, 1: 0.33, 2: 0.67, 3: 1}, 
            "noisy": {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}, 
            "near_random": {0: 0.4, 1: 0.47, 2: 0.53, 3: 0.6}
            }
    
    doc_ids_map = dict(zip(doc_ids_map[:, 0], doc_ids_map[:, 1]))
    
    # Set surgery function mapping
    surgery_functions = {
        'negative': perform_targeted_embedding_surgery_neg,
        'positive': perform_targeted_embedding_surgery_pos,
        'pos_neg': perform_embedding_surgery
    }
    surgery_func = surgery_functions.get(args.surgery_func)
    
    #---------------------------------------------------
    
    query_runs_dict, surgery_runs_dict, tuples, margins, delta_vars_dict, feedback, seconds, metric_evals = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
    updated_embedding = doc_embeddings.copy()
    
    # perform retrieval
    print("Running retrieval before surgery...")
    run = run_faiss_retrieval(index=index, doc_ids=doc_ids, query_embeddings=query_embeddings, query_ids=query_ids, top_k=args.top_k)
    
    # measure scores
    converted_run = {k1: {doc_ids_map.get(int(k2)): v2 for k2, v2 in v1.items()} for k1, v1 in run.items()}
    eval_res = evaluate(qrels, converted_run)
    metric_evals['-1'] = eval_res
    
    print(f'nDCG@10:', np.mean([v.get(f'ndcg_cut_10') for k,v in eval_res.items()]))
    
    # run simulation
    print('Starting surgery')
    for n_qid, qid in enumerate(queries, start=1):
        q_idx = query_ids_map.get(qid) # get the index of the query
        q_emb = query_embeddings[q_idx:q_idx+1] # get the embedding of the query
        
        # retrieve top_k documents for the chosen query
        run_qid = run_faiss_retrieval(index=index, doc_ids=doc_ids, query_embeddings=q_emb, query_ids=[qid], top_k=args.top_k)

        if approach == 'click_log':            
            qid_pairs = click_log(qrels_df, click_distributions, run_qid, args.top_k, args.user_type, args.eta_bias)
        else:            
            retrieved_docs_ids = [elem[0] for elem in sorted(list(run_qid.get(qid).items()), key=lambda x: x[1], reverse=True)]
            
            if approach == 'golden_standard':
                docs_rel = [(doc, qrels.get(qid).get(doc_ids_map.get(int(doc)), -1)) for doc in retrieved_docs_ids]
                reranked_docs = list(sorted(docs_rel, key=lambda x: x[1], reverse=True))
            else:
                retrieved_docs_txt = [doc_lookup.get(doc_ids_map.get(int(doc_id))) for doc_id in retrieved_docs_ids] #get_ids_txt(run_qid, doc_ids_map, doc_lookup)
                reranked_docs = reranker.rank(query=query_lookup.get(qid), docs=retrieved_docs_txt, doc_ids=retrieved_docs_ids, return_scores=True)
                
            reranked_docids = [doc for doc, _ in reranked_docs]
            feedback[qid] = reranked_docs
            
            docid_index = dict(zip(reranked_docids, range(len(reranked_docids))))
            arr = [docid_index.get(x) for x in retrieved_docs_ids]
            _, qid_pairs = bubble_sort(arr.copy(), retrieved_docs_ids.copy())
        
        if qid_pairs:            
            qid_pairs_indexes = [(int(x), int(y)) for (x,y) in qid_pairs]
            margin = np.min([0.01, *[run_qid.get(qid).get(x) - run_qid.get(qid).get(y) for x, y in qid_pairs]])
            
            t0 = time.perf_counter()            
            updated_embedding, indices, delta_vars = surgery_func(query_vec=q_emb, doc_vecs=updated_embedding, pairs=qid_pairs_indexes, margin=margin )
            
            # update the index
            for id_ in indices:
                index.remove_ids(np.array([id_], dtype='int64'))
                index.add_with_ids(updated_embedding[id_].reshape(1, -1), np.array([id_], dtype='int64'))
            elapsed = np.round(time.perf_counter() - t0, 3)
            
            tuples[n_qid] = qid_pairs

        if (approach=='llm' or approach=='golden_standard') or (approach=='click_log' and n_qid%10==0):
            # run
            run = run_faiss_retrieval( index=index, doc_ids=doc_ids, query_embeddings=query_embeddings, query_ids=query_ids, top_k=args.top_k)
            
            run = {k1: {doc_ids_map.get(int(k2)): v2 for k2, v2 in v1.items()} for k1, v1 in run.items()}
            eval_res = evaluate(qrels, run)
            
            metric_evals[n_qid] = eval_res    
            surgery_runs_dict[n_qid] = run
        
            print(f'Query #{n_qid}, nDCG@10:', np.mean([v.get(f'ndcg_cut_10') for _, v in eval_res.items()]))
    
    
    write_json(surgery_runs_dict, results_dir + f'surgery_runs.json.gz')
    write_json(tuples, results_dir + f'negative_positive_pairs.json.gz')
    write_json(metric_evals, results_dir + f'scores.json.gz')
    
if __name__ == "__main__":
    
    args = parse_args()
    
    if Path(args.corpus).is_file(): 
        corpus_name = args.corpus.split('/')[-1].split('.')[0]
    else:
        corpus_name = args.corpus.replace('/', '_')
        
    if Path(args.queries).is_file(): 
        query_name = args.queries.split('/')[-1].split('.')[0]
    else:
        query_name = args.queries.replace('/', '_')
    
    model_name = args.model.replace("/", "_")   
    approach = args.approach.lower()
    
    if approach == 'llm':
        results_dir = f'../results/{corpus_name}/{args.reranker.lower()}/{model_name}/{query_name}/topk_{args.top_k}_surgery_{args.surgery_func}/'
    elif approach == 'golden_standard':
        results_dir = f'../results/{corpus_name}/{approach}/{model_name}/{query_name}/topk_{args.top_k}_surgery_{args.surgery_func}/'
    else:
        results_dir = f'../results/{corpus_name}/{approach}/{model_name}/{query_name}/topk_{args.top_k}_surgery_{args.surgery_func}_nsim_{args.n_sim}_eta_{args.eta_bias}_user_{args.user_type}/'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with open(results_dir+"args_dump.txt", "w") as f:
        for arg_name, arg_value in vars(args).items():
            f.write(f"{arg_name}: {arg_value}\n") 
            
    main(args, results_dir)