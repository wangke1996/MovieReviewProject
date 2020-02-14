# -*- coding: utf-8 -*-

import json
from preprocess import WordSet, WordEmbedding, KnowledgeBase
from sentiment_analysis import SentimentAnalysis
import jieba
import pandas as pd

with open("data.json", "r", encoding="UTF-8") as f_load:
    r_load = json.load(f_load)

#print(r_load)

def frp_multi(fr):
	
		zz = fr['reviews']
		multi_list = []
		
		for yy in zz:
			xx = yy['summary']
			multi_list.append(xx)
		
		abc = SentimentAnalysis()
		list_tds = []
		list_qc = []
		list_c = []
		list_dict = []
		
		for ss in multi_list:
			ss_a = list(jieba.cut(ss,cut_all=False))
			ss_b = " ".join(ss_a)
			#print (ss_b)
			result = abc.analyze(ss_b)
			#print (result)
			for item in result:
				t = item[0]
				d = item[1]
				s = item[4]
				tds = [t,d,s]
				list_tds.append(tds)
			
		for tri in list_tds:
			if tri not in list_qc:
				list_qc.append(tri)
				tri_count = list_tds.count(tri)
				list_c.append(tri_count)
		
		for ww in list_qc:
			vc_index = list_qc.index(ww)
			vc = list_c[vc_index]
			vt = ww[0]
			vd = ww[1]
			vs = ww[2]
			dict_a = {"对象":vt,"评价极性":vs,"描述词":vd,"评论数":vc}
			list_dict.append(dict_a)
		
		df = pd.DataFrame(list_dict,columns=["对象","评价极性","描述词","评论数"])
		df.to_csv("./ndetails.csv",index=False)


frp_multi(r_load)
				
				
				
				
				
				
				
				
				
				
				
