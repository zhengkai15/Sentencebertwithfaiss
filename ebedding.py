from sentence_transformers import SentenceTranformer, util
import os 
import csv
import pickle
import time
import faiss
import numpy as np
import json

model_name = 'chinese_wwm_ext_L-12_H-768_A-12'
model = SentenceTransformer(model_name)

dataset_path = "train.jsonl"
max_corpus_size = 100000
embedding_cache_path = 'jsonl-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)
enbedding_size = 768
top_k_hits = 10

#Defining our Faiss index
n_clusters = 1024
quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
index.nprobe = 3

#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(dataset_path, encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            corpus_sentences.add(row[''])
            if len(corpus_sentences) >= max_corpus_size:
                break

            corpus_sentences.add(row[''])
            if len(corpus_sentences) >= max_corpus_size:
                break

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']









