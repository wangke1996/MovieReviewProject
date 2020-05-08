from backend.sentiment.sentiment_analysis import SentimentAnalysis
from backend.functionLib.function_lib import concat_list, cut_sentences, split_sentences, read_lines, file_hash, \
    load_json_file, save_json_file
from backend.config import CONFIG
import os


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

    def analysis_single_review(self, review):
        sentences = split_sentences(review)
        cut_sentences_list = cut_sentences(sentences)
        triples = concat_list(map(self.analysis_single_sentence, cut_sentences_list))

        predict_score = round(5 * len([x for x in triples if x['sentiment'] != 'NEG']) / len(triples))
        return {'data': triples, 'score': predict_score}

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

    def analysis_reviews(self, review_list):
        sentences = concat_list(map(lambda x: split_sentences(x), review_list))
        cut_sentences_list = cut_sentences(sentences)
        details = self.analysis_multi_sentences(cut_sentences_list)
        return details

    def analysis_uploaded_file(self, file_name):
        file = os.path.join(CONFIG.upload_folder, file_name)
        md5 = file_hash(file)
        cache = md5.hexdigest()
        cache_file = os.path.join(CONFIG.upload_analysis_cache_folder, cache + '.json')
        if os.path.exists(cache_file):
            return load_json_file(cache_file), cache
        reviews = read_lines(file, lambda x: x.strip())
        details = self.analysis_reviews(reviews)
        save_json_file(cache_file, details)
        return details, cache


sentimentAnalyzer = SentimentAnalyzer()
