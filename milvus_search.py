from sentence_transformers import SentenceTranformer, util
import os
import csv
import pickle
import time
import faiss
import numpy as np
import json
if __name__ == '__main__':

def milvus_search(str_query)
	model_name = 'chinese_wwm_ext_L-12_H-768_A-12'
	model = SentenceTransformer(model_name)

	dataset_path = "train.jsonl"
	max_corpus_size = 100000
	embedding_cache_path = 'jsonl-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)
	enbedding_size = 768
	# top_k_hits = 10


	#Check if embedding cache path exists
	# if not os.path.exists(embedding_cache_path):
	#     # Check if the dataset exists. If not, download and extract
	#     # Download dataset if needed
	#     if not os.path.exists(dataset_path):
	#         print("Download dataset")
	#         util.http_get(url, dataset_path)

	#     # Get all unique sentences from the file
	#     corpus_sentences = set()
	#     with open(dataset_path, encoding='utf8') as fIn:
	#         reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
	#         for row in reader:
	#             corpus_sentences.add(row[''])
	#             if len(corpus_sentences) >= max_corpus_size:
	#                 break

	#             corpus_sentences.add(row[''])
	#             if len(corpus_sentences) >= max_corpus_size:
	#                 break

	#     corpus_sentences = list(corpus_sentences)
	#     print("Encode the corpus. This might take a while")
	#     corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

	#     print("Store file on disc")
	#     with open(embedding_cache_path, "wb") as fOut:
	#         pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
	# else:
	#     print("Load pre-computed embeddings from disc")
	#     with open(embedding_cache_path, "rb") as fIn:
	#         cache_data = pickle.load(fIn)
	#         corpus_sentences = cache_data['sentences']
	#         corpus_embeddings = cache_data['embeddings']

	print("Load pre-computed embeddings from disc")
	with open(embedding_cache_path, "rb") as fIn:
	    cache_data = pickle.load(fIn)
	    corpus_sentences = cache_data['sentences']
	    corpus_embeddings = cache_data['embeddings']



	import numpy as np
	from milvus import Milvus, IndexType, MetricType

	# 初始化一个Milvus类，以后所有的操作都是通过milvus来的
	host = 'localhost'
	port = '19530'
	milvus = Milvus(host,port)
	collection_name = "col_2"
	# 连接到服务器，注意端口映射，要和启动docker时设置的端口一致


	# 向量个数
	num_vec = 16370
	# 向量维度
	vec_dim = 768

	collection_param = {
	    'collection_name': collection_name,
	    'dimension':vec_dim,
	    'index_file_size':1024,
	    'metric_type':MetricType.IP}
	milvus.create_collection(collection_param)

	id = [i for i in range(16370)]
	status,ids= milvus.insert(collection_name,corpus_embeddings,id)

	index_param = {
	    'nlist': 128
	    }
	milvus.create_index(collection_name,IndexType.FLAT index_param)


	# query_vectors =corpus_embeddings[0:1]
	query_vectors1= model.encode(str_query,convert_to_tensor= True)
	query_vectors = query_vectors1.reshape(-1,1).transpose(0,1).tolist()

	search_param={
	    'nprob' = 16
	}

	print("searching")


	params ={
	    collection_name:collection_name,
	    'query_record':query_vectors,
	    'top_k':1,
	    'params':search_param
	}
	status,results = milvus.search(**params)
	if status.OK():
	    if results[0][0].distance == 0.0 or results[0][0].id == ids[0]:
	        print('Query result is correct')
	    else:
	        print('Query result isn\'t correct')

	   # print results
	    # print(results)
	    count = -1
	    for score,idx in zip(results.distance_array,results.id_array):
	    	count += 1
	    	print(score,idx)
	    	for i in range(len(score)):
	    		print(count,'.',i,corpus_sentences(idx[i],"(Score:{:.4f})".format(score[i])))
	else:
	    print("Search failed. ", status)

	        # Delete demo_collection
	status = milvus.drop_collection(collection_name)

if __name__ == '__main__':
	milvus_search("你好")







