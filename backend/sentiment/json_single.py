# -*- coding: utf-8 -*-

import json
from preprocess import WordSet, WordEmbedding, KnowledgeBase
from sentiment_analysis import SentimentAnalysis

with open('sample.txt','r',encoding='UTF-8') as f:
	sentence = f.read()

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def frp_single(sentence):
	
	abc = SentimentAnalysis()
	result = abc.analyze(sentence)
	str_a = []
	jsonlist_a = []
	
	for item in result:
		aspect = item[0]
		opinion = item[1]
		relation = item[4]
		#t = ***
		a = {'target':aspect,'description':opinion,'sentiment':relation}
		str_a.append(a)
	
	for  i in str_a:
		json_info = json.dumps(i,default=set_default,ensure_ascii=False)
		jsonlist_a.append(json_info)
	
	return jsonlist_a

xx = frp_single(sentence)
print(xx)
