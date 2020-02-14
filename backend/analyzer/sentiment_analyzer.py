from backend.sentiment.sentiment_analysis import SentimentAnalysis
from backend.functionLib.function_lib import concat_list


class SentimentAnalyzer(object):
    def __init__(self):
        self.analyzer = SentimentAnalysis()

    def analysis_single_sentence(self, cut_sentence):
        try:
            result = self.analyzer.analyze(cut_sentence)
        except:
            return []
        triples = list(map(
            lambda item: {'target': item[0], 'description': item[1], 'sentiment': item[4], 'sentence': cut_sentence},
            result))
        return triples

    def analysis_multi_sentences(self, cut_sentences_list):
        triples = concat_list(map(self.analysis_single_sentence, cut_sentences_list))
        result = {}
        for item in triples:
            target = item['target']
            description = item['description']
            sentiment = item['sentiment']
            sentence = item['sentence']
            if target not in result:
                result[target] = {'POS': {}, 'NEU': {}, 'NEG': {}}
            if description not in result[target][sentiment]:
                result[target][sentiment][description] = []
            result[target][sentiment][description].append(sentence)
        return result


sentimentAnalyzer = SentimentAnalyzer()
