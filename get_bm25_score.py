from gensim.summarization import bm25
from transformers import BertTokenizer
def get_bm25(dis_name,dia_name):
	tockenizer = BertTokenizer.from_pretrained(' ',do_lower_case=True)
	x1=tockenizer.tockenize(str(dis_name))
	bm25Model = bm25.BM25(x1)
	scores= bm25Model.get_scores(dia_name)
	scores = max(scores)
	return scores