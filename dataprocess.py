import json
import os
import time
import jsonlines


def qure_prossed(path_in,path_out):
    with jsonlines.open(path_in,'r') as f:
        data_new = ""
        for item in f :
            qurey = item["query"]
            qurey_id = item["query_id"]
            doc = item["doc"]
            doc_id = item["doc_id"]
            label = item["label"]
            retrieval_score = item["retrieval_score"]
            data_new += str(str(qurey)+"\t"+str(qurey_id)+"\t"+str(doc)+"\t"+str(doc_id)+"\t"+str(label)+"\t"+str(retrieval_score)+"/n")
        with open(path_out,'w',encoding = 'utf-8') as g:
            g.write(data_new)
if __name__ == "__main__":
    path = './'
    path_in = os.path.join(path,'train.jsonl')
    path_out =os.path.join(path,'train.tsv')
    qure_prossed(path_in,path_out)




