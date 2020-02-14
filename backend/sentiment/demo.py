# -*- coding: utf-8 -*-

# sentence: 需要分好词 空格分隔 2<=词语数量<=30
# result: (aspect, opinion, aspect_index, opinion_index, relation, knowledge)
# knowledge: (aspect, opinion, relation, weight)

import argparse

from preprocess import WordSet, WordEmbedding, KnowledgeBase
from sentiment_analysis import SentimentAnalysis


#sentence = '外观 漂亮'
#sentence = '外观 不 太 漂亮'
#sentence = '高 规格 的 用料 和 精致 的 做工'
#sentence = '炫酷 的 造型 、 充沛 的 动力 再 加上 本田 家族 运动 基因 的 传承'

parser = argparse.ArgumentParser()
parser.add_argument('-s', required=True)
args = parser.parse_args()

sentence = args.s

abc = SentimentAnalysis()
result = abc.analyze(sentence)


print('--------------------')
print('%s\n' % (sentence))
for item in result:
    aspect = item[0]
    opinion = item[1]
    relation = item[4]
    print('%s\t%s\t%s' % (aspect, opinion, relation))
print('--------------------')

for item in result:
    aspect = item[0]
    opinion = item[1]
    relation = item[4]
    knowledge = item[5]
    knowledge = sorted(knowledge, key=lambda t: t[3], reverse=True)
    print('--------------------')
    print('%s\t%s\t%s\n' % (aspect, opinion, relation))
    for k in knowledge:
        print('%s\t%s\t%s\t%f' % (k[0], k[1], k[2], k[3]))
    print('--------------------')   


