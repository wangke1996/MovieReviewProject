from backend.crawler.douban_crawler import crawler
from backend.config import config
from backend.functionLib.function_lib import load_json_file, save_json_file, cache_available, split_sentences, \
    concat_list, cut_sentences, search_candidate
from .sentiment_analyzer import sentimentAnalyzer
from time import sleep

import os

max_review_count = 10000


class DataAnalyzer:
    def __init__(self):
        self.details = {}  # {target1:{'POS':{description1:[sentence1,sentence2],description2:[]},'NEU':{},'NEG:{}},target2:{}}
        self.target_freq = {}
        self.movie_id = None
        self.lock = {}

    def analyzeReviewsTrend(self, movie_id):
        # 上锁避免重复分析
        while self.lock.get(movie_id, False):
            sleep(5)
        folder = os.path.join(config.data_path, 'subject', movie_id, 'analysis')
        os.makedirs(folder, exist_ok=True)
        json_file = os.path.join(folder, 'reviewsTrend.json')
        if cache_available(json_file, update_interval=-1):
            results = load_json_file(json_file)
        else:
            self.lock[movie_id] = True
            reviews = crawler.get_movie_reviews(movie_id, reviews_count=max_review_count)
            results = {}
            for review in reviews:
                create_time = review["created_at"].split()[0]
                rate = review['rating']['value']
                if create_time not in results:
                    results[create_time] = {'num': 0, 'rate': 0}
                results[create_time]['num'] = results[create_time]['num'] + 1
                results[create_time]['rate'] = results[create_time]['rate'] + rate
            results = [{'time': x[0], 'num': x[1]['num'], 'rate': x[1]['rate'] / x[1]['num']} for x in results.items()]
            results.sort(key=lambda d: tuple(map(int, d['time'].split('-'))))  # sort by date
            save_json_file(json_file, results)
            self.lock[movie_id] = False
        # print(results)
        return results

    def get_target_freqs(self, movie_id, target):
        _, target_freq = self.analyzeMovieProfile(movie_id)
        if target not in target_freq:
            return {}
        return target_freq[target]

    def analyzeMovieProfile(self, movie_id):
        if self.movie_id is not None and self.movie_id == movie_id:
            return self.details, self.target_freq
        folder = os.path.join(config.data_path, 'subject', movie_id, 'analysis')
        os.makedirs(folder, exist_ok=True)
        json_file = os.path.join(folder, 'profile.json')
        if cache_available(json_file, update_interval=-1):
            details = load_json_file(json_file)
        else:
            reviews = crawler.get_movie_reviews(movie_id, reviews_count=max_review_count)
            sentences = concat_list(map(lambda x: split_sentences(x['content']), reviews))
            cut_sentences_list = cut_sentences(sentences)
            details = sentimentAnalyzer.analysis_multi_sentences(cut_sentences_list)
            save_json_file(json_file, details)
        target_freq = {}
        for target, item in details.items():
            freq_dict = {}
            for sentiment, descriptions in item.items():
                freq_dict[sentiment] = sum(map(len, descriptions.values()))
            freq_dict['freq'] = sum(freq_dict.values())
            target_freq[target] = freq_dict
        self.movie_id = movie_id
        self.details = details
        self.target_freq = target_freq
        return self.details, self.target_freq

    def get_target_list(self, movie_id, count=10, sort_by='freq', min_freq=50):
        _, target_freq = self.analyzeMovieProfile(movie_id)
        target_list = list(filter(lambda x: x[1]['freq'] >= min_freq, target_freq.items()))
        if sort_by == 'freq':
            # 按频率排序
            target_list.sort(key=lambda x: x[1][sort_by], reverse=True)
        else:
            # sort_by== 'POS' or 'NEG'
            # 按好评率、差评率排序
            target_list.sort(key=lambda x: x[1][sort_by] / x[1]['freq'], reverse=True)
        target_list = target_list[:count]
        top_targets = list(map(lambda x: {'name': x[0], **x[1]}, target_list))
        return top_targets

    def get_target_detail(self, movie_id, target):
        details, target_freq = self.analyzeMovieProfile(movie_id)
        descriptions = details[target]
        result = {}
        for k, v in descriptions.items():
            result[k] = list(map(lambda x: {'name': x[0], 'freq': len(x[1]), 'sentiment': k}, v.items()))
            result[k].sort(key=lambda x: x['freq'], reverse=True)
        return result

    def get_related_sentences(self, movie_id, target, sentiment, description, start_index, count):
        details, _ = self.analyzeMovieProfile(movie_id)
        sentences = details[target][sentiment][description]
        sentence_num = len(sentences)
        data = sentences[start_index:start_index + count]
        new_start_index = min(start_index + count, sentence_num)
        loaded_all = new_start_index == sentence_num
        return {'loadedAll': loaded_all, 'startIndex': new_start_index, 'data': data}

    def search_target(self, movie_id, input_value, search_mode='char'):
        _, target_freq = self.analyzeMovieProfile(movie_id)
        targets = list(target_freq.keys())
        candidates = search_candidate(targets, input_value, mode=search_mode)
        return list(map(lambda x: {'name': x, **target_freq[x]}, candidates))


dataAnalyzer = DataAnalyzer()
