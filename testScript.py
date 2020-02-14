# from backend.sentiment.preprocess import WordSet, WordEmbedding, KnowledgeBase
# from backend.sentiment import model_CNN, model_MNKG
# import sys
#
# sys.modules['model_CNN'] = model_CNN
# sys.modules['model_MNKG'] = model_MNKG
from backend.sentiment import *
from backend.analyzer import sentimentAnalyzer

sentence = '炫酷 的 造型 、 充沛 的 动力 再 加上 本田 家族 运动 基因 的 传承'
# sentence = '外观 漂亮'
result = sentimentAnalyzer.analysis_single_sentence(sentence)
print(result)
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
