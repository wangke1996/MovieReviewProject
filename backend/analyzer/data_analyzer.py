from backend.crawler.douban_api_crawler import api_crawler
from backend.config import CONFIG
from backend.functionLib.function_lib import load_json_file, save_json_file, cache_available, split_sentences, \
    concat_list, cut_sentences, search_candidate, load_np_array
from .sentiment_analyzer import sentimentAnalyzer
from .user_analyzer import userAnalyzer
from time import sleep

import os

max_review_count = 10000
max_comment_count = 500


class DataAnalyzer:
    def __init__(self):
        self.details = {}  # {target1:{'POS':{description1:[sentence1,sentence2],description2:[]},'NEU':{},'NEG:{}},target2:{}}
        self.target_freq = {}
        self._id = None
        self._type = 'movie'
        self.lock = {'movie': {}, 'user': {}, 'cache': {}}

    def wait_for_lock(self, _type=None, _id=None):
        _type = _type or self._type or 'movie'
        _id = _id or self._id or ''
        while self.lock[_type].get(_id, False):
            sleep(5)

    def set_lock(self, _type, _id):
        _type = _type or self._type or 'movie'
        _id = _id or self._id or ''
        self.lock[_type][_id] = True

    def free_lock(self, _type, _id):
        _type = _type or self._type or 'movie'
        _id = _id or self._id or ''
        self.lock[_type][_id] = False

    @staticmethod
    def get_folder(_id, _type, folder_name='analysis', make=True):
        folder = os.path.join(CONFIG.data_path, 'subject' if _type in ['movie', 'subject'] else 'user', _id,
                              folder_name)
        if make:
            os.makedirs(folder, exist_ok=True)
        return folder

    def analyze_profile(self, _id, _type, reviews_count=max_review_count, comments_count=max_comment_count):
        if self._id is not None and self._id == _id and self._type == _type:
            return self.details, self.target_freq
        self.wait_for_lock(_type, _id)
        if _type == 'cache':
            profile_file = os.path.join(CONFIG.upload_analysis_cache_folder, '%s.json' % _id)
            details = load_json_file(profile_file)
        else:
            folder = self.get_folder(_id, _type)
            profile_file = os.path.join(folder, 'profile.json')
            if cache_available(profile_file, update_interval=-1):
                details = load_json_file(profile_file)
            else:
                self.set_lock(_type, _id)
                self.analyze_reviews(_id, _type, reviews_count)
                self.analyze_comment(_id, _type, comments_count)
                details = self.merge_review_comment_profiles(_id, _type)
                self.free_lock(_type, _id)
        target_freq = {}
        for target, item in details.items():
            freq_dict = {}
            for sentiment, descriptions in item.items():
                freq_dict[sentiment] = sum(map(len, descriptions.values()))
            freq_dict['freq'] = sum(freq_dict.values())
            target_freq[target] = freq_dict
        self._id = _id
        self._type = _type
        self.details = details
        self.target_freq = target_freq
        return self.details, self.target_freq

    def analyze_reviews(self, _id, _type, reviews_count=max_review_count):
        folder = self.get_folder(_id, _type)
        json_file = os.path.join(folder, 'profile_review.json')
        if cache_available(json_file, update_interval=-1):
            details = load_json_file(json_file)
        else:
            if _type == 'movie':
                reviews = api_crawler.get_movie_reviews(_id, reviews_count=reviews_count)
            else:
                reviews = userAnalyzer.get_reviews(_id)
            details = sentimentAnalyzer.analysis_reviews([x['content'] for x in reviews])
            save_json_file(json_file, details)
        return details

    def analyze_comment(self, _id, _type, comments_count=max_comment_count):
        folder = self.get_folder(_id, _type)
        json_file = os.path.join(folder, 'profile_comment.json')
        if cache_available(json_file, update_interval=-1):
            details = load_json_file(json_file)
        else:
            if _type == 'movie':
                comments = api_crawler.get_movie_comments(_id, comments_count)
                comments = [x['content'] for x in comments]
            else:
                comments = userAnalyzer.get_collect(_id)
                comments = [x['comment'] for x in comments]
            details = sentimentAnalyzer.analysis_reviews(comments)
            save_json_file(json_file, details)
        return details

    def analyze_movie_profile(self, movie_id, reviews_count=max_review_count, comments_count=max_comment_count):
        return self.analyze_profile(movie_id, 'movie', reviews_count, comments_count)

    def analyze_movie_reviews(self, movie_id, reviews_count=max_review_count):
        return self.analyze_reviews(movie_id, 'movie', reviews_count)

    def analyze_movie_comments(self, movie_id, comments_count=max_comment_count):
        return self.analyze_comment(movie_id, 'movie', comments_count)

    def analyze_user_reveiws(self, user_id):
        return self.analyze_reviews(user_id, 'user')

    def analyze_user_comments(self, user_id):
        return self.analyze_comment(user_id, 'user')

    def merge_review_comment_profiles(self, _id, _type='subject'):
        folder = self.get_folder(_id, _type)
        merged_profile_file = os.path.join(folder, 'profile.json')
        if os.path.exists(merged_profile_file):
            return load_json_file(merged_profile_file)
        os.makedirs(folder, exist_ok=True)
        review_profile_file = os.path.join(folder, 'profile_review.json')
        comment_profile_file = os.path.join(folder, 'profile_comment.json')
        if os.path.exists(review_profile_file):
            review_profile = load_json_file(review_profile_file)
        else:
            review_profile = {}
        if os.path.exists(comment_profile_file):
            comment_profile = load_json_file(comment_profile_file)
        else:
            comment_profile = {}
        profile = {}
        targets = set(list(review_profile.keys()) + list(comment_profile.keys()))
        sentiments = ["POS", "NEU", "NEG"]
        for target in targets:
            profile[target] = {}
            for sentiment in sentiments:
                profile[target][sentiment] = {}
                descriptions = set(list(review_profile.get(target, {}).get(sentiment, {}).keys()) +
                                   list(comment_profile.get(target, {}).get(sentiment, {}).keys()))
                for description in descriptions:
                    profile[target][sentiment][description] = \
                        review_profile.get(target, {}).get(sentiment, {}).get(description, []) + \
                        comment_profile.get(target, {}).get(sentiment, {}).get(description, [])
        save_json_file(merged_profile_file, profile)
        return profile

    def analyze_movie_reviews_trend(self, movie_id):
        # 上锁避免重复分析
        self.wait_for_lock('movie', movie_id)
        folder = os.path.join(CONFIG.data_path, 'subject', movie_id, 'analysis')
        os.makedirs(folder, exist_ok=True)
        json_file = os.path.join(folder, 'reviewsTrend.json')
        if cache_available(json_file, update_interval=-1):
            results = load_json_file(json_file)
        else:
            self.set_lock('movie', movie_id)
            reviews = api_crawler.get_movie_reviews(movie_id, reviews_count=max_review_count)
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
            self.free_lock('movie', movie_id)
        # print(results)
        return results

    def get_target_freqs(self, _id, _type, target):
        _, target_freq = self.analyze_profile(_id, _type)
        if target not in target_freq:
            return {}
        return target_freq[target]

    def get_target_list(self, _id, _type, count=10, sort_by='freq', min_freq=0):
        _, target_freq = self.analyze_profile(_id, _type)
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

    def get_target_detail(self, _id, _type, target):
        details, target_freq = self.analyze_profile(_id, _type)
        descriptions = details[target]
        result = {}
        for k, v in descriptions.items():
            result[k] = list(map(lambda x: {'name': x[0], 'freq': len(x[1]), 'sentiment': k}, v.items()))
            result[k].sort(key=lambda x: x['freq'], reverse=True)
        return result

    def get_related_sentences(self, _id, _type, target, sentiment, description, start_index, count):
        details, _ = self.analyze_profile(_id, _type)
        sentences = details[target][sentiment][description]
        sentence_num = len(sentences)
        data = sentences[start_index:start_index + count]
        new_start_index = min(start_index + count, sentence_num)
        loaded_all = new_start_index == sentence_num
        return {'loadedAll': loaded_all, 'startIndex': new_start_index, 'data': data}

    def search_target(self, _id, _type, input_value, search_mode='char'):
        _, target_freq = self.analyze_profile(_id, _type)
        targets = list(target_freq.keys())
        candidates = search_candidate(targets, input_value, mode=search_mode)
        return list(map(lambda x: {'name': x, **target_freq[x]}, candidates))

    @staticmethod
    def get_user_info(user):
        info = userAnalyzer.get_basic_info(user)['info']
        info.update({'id': user})
        return info

    def get_active_users(self, num=10):
        users = list(load_np_array(CONFIG.user_list_file))

        def get_sentiment_num(user):
            _, freqs = self.analyze_profile(user, 'user')
            total_freq = sum([x.get('freq', 0) for x in freqs.values()])
            return total_freq

        users.sort(key=get_sentiment_num, reverse=True)
        return list(map(self.get_user_info, users[:num]))

    def search_user(self, query):
        users = list(load_np_array(CONFIG.user_list_file))
        infos = []
        for user in users:
            info = self.get_user_info(user)
            if query in info['name'] or query in info['id']:
                infos.append(info)
        return infos

    def check_user_state(self, uid):
        profile_file = os.path.join(self.get_folder(uid, 'user'), 'profile.json')
        if os.path.exists(profile_file):
            return 'ok'
        else:
            user_info = api_crawler.get_user_info(uid, update_interval=-1)
            if len(user_info) == 0 or user_info["is_suicide"]:
                return 'error'
            else:
                return 'uncached'


dataAnalyzer = DataAnalyzer()
