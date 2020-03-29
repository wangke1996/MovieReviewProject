import os
import sys
import re
import time
import numpy as np
import itertools
from collections import Counter
import json
from backend.crawler.douban_crawler import douban_crawler
from lxml import etree
from backend.crawler.douban_api_crawler import api_crawler
from backend.config import CONFIG
from backend.functionLib.function_lib import load_json_file, save_json_file, cache_available, load_np_array, \
    logging_with_time, clean_error_crawl
from backend.crawler.limit import Limit
import random


class UserAnalyzer(object):
    def __init__(self):
        self.uid = None
        self.user_folder = None
        self.collect_file = None
        pass

    def update_uid(self, uid):
        if self.uid is None or uid != self.uid:
            self.uid = uid
            self.user_folder = os.path.join(CONFIG.user_path, uid)
            self.info_file = os.path.join(self.user_folder, 'info.json')
            self.collect_folder = os.path.join(self.user_folder, 'collect')
            self.collect_file = os.path.join(self.collect_folder, 'collect.json')
            self.collect_html_file = os.path.join(self.collect_folder, 'html.json')
            self.review_folder = os.path.join(self.user_folder, 'review')
            self.review_file = os.path.join(self.review_folder, 'review.json')
            self.review_list_html_file = os.path.join(self.review_folder, 'review_list_html.json')
            self.review_html_file = os.path.join(self.review_folder, 'review_html.json')
            self.profile_folder = os.path.join(self.user_folder, 'profile')
            self.basic_info_file = os.path.join(self.profile_folder, 'basic_info.json')
            self.collect_distribution_file = os.path.join(self.profile_folder, 'collect_distribution.json')
            self.sentiment_profile_file = os.path.join(self.profile_folder, 'sentiment_profile.json')
            os.makedirs(self.user_folder, exist_ok=True)
            os.makedirs(self.collect_folder, exist_ok=True)
            os.makedirs(self.review_folder, exist_ok=True)
            os.makedirs(self.profile_folder, exist_ok=True)
            # clean_error_crawl(self.user_folder)

    def get_collection_tags(self, uid):
        url = 'https://movie.douban.com/j/people/%s/get_collection_tags' % uid
        try:
            tags = json.loads(Limit.get_request(url, use_proxies=True), encoding='utf8')
            tags.sort(key=lambda x: x["count"], reverse=True)
        except Exception as e:
            print('error in crawl tags: %s' % url)
            print(e)
            tags = []
        return tags

    def get_basic_info(self, uid, update_interval=-1):
        self.update_uid(uid)
        if cache_available(self.basic_info_file, update_interval):
            return load_json_file(self.basic_info_file)
        collects, htmls = self.get_collect(uid, update_interval, True)
        s = etree.HTML(htmls[0]['content'])
        user_name = s.xpath('//div[@class="side-info-txt"]/h3/text()')[0]
        home_page = 'https://movie.douban.com/people/%s' % uid
        avatar = s.xpath('//a[@class="side-info-avatar"]/img/@src')[0]
        total_num = len(collects)
        basic_info = {'info': {'name': user_name, 'avatar': avatar, 'totalNum': total_num, 'homePage': home_page}}
        save_json_file(self.basic_info_file, basic_info)
        return basic_info

    def get_collect_distribution(self, uid, update_interval=-1):
        self.update_uid(uid)
        if cache_available(self.collect_distribution_file, update_interval):
            return load_json_file(self.collect_distribution_file)
        collects = self.get_collect(uid, update_interval)
        reviews = self.get_reviews(uid, update_interval)
        rates = dict(Counter([x["rate"] for x in collects] + [x["rate"] for x in reviews]))
        watched_movies = list(set([x["movie_id"] for x in collects] + [x["movie"] for x in reviews]))
        pubyears = []
        types = []
        casts = []
        directors = []
        countries = []
        for movie in watched_movies:
            movie_info = api_crawler.get_movie_info(movie, update_interval)
            if len(movie_info) == 0:
                print('error in get movie info: %s' % movie)
                continue
            try:
                pubyear = int(movie_info["pubdates"][0][:4])
            except IndexError:
                pubyear = int(([x['date'] for x in collects if x['movie_id'] == movie] or [x['date'] for x in reviews if
                                                                                           x['movie'] == movie])[0][:4])
                print('no pubdate for movie %s, use comment year %d instead' % (movie, pubyear))
            pubyears.append(pubyear)
            types.extend(movie_info["genres"])
            casts.extend([x["id"] for x in movie_info["casts"]])
            directors.extend([x["id"] for x in movie_info["directors"]])
            countries.extend(movie_info["countries"])
        types = dict(Counter(types))
        directors = dict(Counter(directors))
        casts = dict(Counter(casts))
        pubyears = dict(Counter(pubyears))
        countries = dict(Counter(countries))
        tag_distribution = self.get_collection_tags(uid)
        tags = dict([(x['tag'], x['count']) for x in tag_distribution])
        collect_distribution = {'rate': rates, 'type': types, 'director': directors, 'cast': casts, 'pubyear': pubyears,
                                'country': countries, 'tag': tags}

        save_json_file(self.collect_distribution_file, collect_distribution)
        return collect_distribution

    def get_profile_of_collect(self, uid):
        self.update_uid(uid)
        collect_distribution = self.get_collect_distribution(uid)

        def make_distribution(counter: dict, key_name, val_name, key_map=lambda x: x, key_filter=lambda x: True,
                              val_filter=lambda x: True, sort_by=None, reverse=False):
            res = [{key_name: key_map(k), val_name: v} for k, v in counter.items() if
                   key_filter(k) and val_filter(v)]
            keys = set([x[key_name] for x in res])
            merged_res = [{key_name: k, val_name: sum([x[val_name] for x in res if x[key_name] == k])} for k in keys]
            if sort_by is not None:
                if sort_by == 'val':
                    merged_res.sort(key=lambda x: x[val_name], reverse=reverse)
                else:
                    merged_res.sort(key=lambda x: x[key_name], reverse=reverse)
            return merged_res

        rate_distribution = make_distribution(collect_distribution['rate'], 'rate', 'reviewNums',
                                              key_map=lambda x: "%s star" % x, key_filter=lambda x: int(x) > 0,
                                              sort_by='key')
        rate_nums = [(int(rate), num) for rate, num in collect_distribution['rate'].items() if int(rate) > 0]
        average_score = sum([x * y for x, y in rate_nums]) / sum([y for _, y in rate_nums])
        type_distribution = make_distribution(collect_distribution['type'], 'type', 'num', sort_by='val', reverse=True)
        max_type = 12
        type_distribution = type_distribution[:max_type]
        type_distribution.sort(key=lambda x: x['type'])

        def age_map(years, min_age=1960, interval=10):
            min_year = min(years)
            max_year = max(years)
            keys = []
            if min_year < min_age:
                bins = [min_year, min_age]
                keys.append('?~%d' % min_age)
            else:
                bins = [min_age]
            current_age = bins[-1]
            while current_age < max_year:
                bins.append(current_age + interval)
                keys.append('%d~%d' % (current_age, current_age + interval))
                current_age = current_age + interval
            hist, _ = np.histogram(years, bins)
            hist = [int(x) for x in hist]
            assert len(hist) == len(keys)
            return dict(zip(keys, hist))

        pubyears = [int(x) for x in itertools.chain.from_iterable(
            [[year] * num for year, num in collect_distribution['pubyear'].items()])]
        # age_distribution = make_distribution(age_map(pubyears), 'age', 'movieNums')
        age_distribution = make_distribution(collect_distribution['pubyear'], 'year', 'count', key_map=int,
                                             sort_by='key')

        def get_favorite(counter: dict):
            max_val = 0
            favorite = None
            for k, v in counter.items():
                if k == 'null' or k is None:
                    continue
                if v > max_val:
                    max_val = v
                    favorite = k
            return favorite, max_val

        favorite_actor, saw = get_favorite(collect_distribution["cast"])
        actor_info = api_crawler.get_celebrity_info(favorite_actor)
        favorite_actor = {'id': favorite_actor, 'name': actor_info['name'], 'saw': saw, 'url': actor_info['alt'],
                          'img': actor_info["avatars"]["large"], 'description': actor_info['summary']}
        profile_of_collect = {'distribution': {'rateDistribution': rate_distribution, 'averageScore': average_score,
                                               'typeDistribution': type_distribution,
                                               'ageDistribution': age_distribution,
                                               'pubyear': pubyears},
                              'favorite': {'favoriteActor': favorite_actor}}
        return profile_of_collect

    def get_profile_of_sentiment(self, uid, profile):
        target_freq = {}
        for target, item in profile.items():
            freq_dict = {}
            for sentiment, descriptions in item.items():
                freq_dict[sentiment] = sum(map(len, descriptions.values()))
            freq_dict['freq'] = sum(freq_dict.values())
            target_freq[target] = freq_dict
        max_negative_rate = 0.0
        max_negative_num = 0
        worst_target = None
        related_reviews = []
        freq_th = 10
        for target, freqs in target_freq.items():
            total_freq = freqs['freq']
            if total_freq < freq_th:
                continue
            negative_num = freqs['NEG']
            negative_rate = negative_num / total_freq
            if negative_rate > max_negative_rate:
                max_negative_rate = negative_rate
                max_negative_num = negative_num
                worst_target = target
        if worst_target is None:
            # in case that all freqs are lower than th
            for target, freqs in target_freq.items():
                total_freq = freqs['freq']
                negative_num = freqs['NEG']
                negative_rate = negative_num / total_freq
                if negative_num > max_negative_num:
                    max_negative_num = negative_num
                    max_negative_rate = negative_rate
                    worst_target = target
        if worst_target is None:
            # in case that none negative sentiment found
            print('no negative sentiment profile found for user %s' % uid)
        else:
            # sort descriptions to worst target
            descirption_sentences = list(profile[worst_target]['NEG'].items())
            descirption_sentences.sort(key=lambda x: len(x[1]), reverse=True)
            for description, sentences in descirption_sentences:
                sentences = list(set(sentences))
                random.shuffle(sentences)
                review = None
                for sentence in sentences:
                    review = self.find_review_of_sentence(sentence.replace(' ', '').strip())
                    if review is not None:
                        s = sentence.replace(' ', '').strip()
                        sentence_index = review['content'].index(s)
                        sentence_medium = sentence_index + len(s) // 2
                        target_indexes = [m.start() for m in re.finditer(worst_target, review["content"])]
                        target_indexes.sort(key=lambda x: abs(x + len(worst_target) // 2 - sentence_medium))
                        try:
                            target_index = target_indexes[0]
                        except IndexError:
                            target_index = -1
                        description_indexes = [m.start() for m in re.finditer(description, review["content"])]
                        description_indexes.sort(key=lambda x: abs(x + len(description) // 2 - sentence_medium))
                        try:
                            description_index = description_indexes[0]
                        except IndexError:
                            description_index = -1
                        review.update({'target': worst_target, 'description': description, 'targetIndex': target_index,
                                       'descriptionIndex': description_index})
                        if target_index == -1 or description_index == -1:
                            print('cannot get the index of target of description in %s' % (str(review)))
                            print('sentence: %s' % s)
                        movie_id = review["movie_id"]
                        movie_info = api_crawler.get_movie_info(movie_id)
                        if len(movie_info) == 0:
                            print('cannot get movie info of %s' % movie_id)
                            continue
                        review.update({"movie": movie_info["title"], "movie_img": movie_info["images"]["large"],
                                       "url": '/movieProfile/%s' % movie_id})
                        related_reviews.append(review)
                        break
                if review is None:
                    continue
        return {
            'sentiment': {'reviewList': related_reviews, 'worstTarget': worst_target, 'negativeRate': max_negative_rate,
                          'negativeNum': max_negative_num}}

    def find_review_of_sentence(self, sentence):
        collects = self.get_collect(self.uid)
        res = None
        for collect in collects:
            if sentence in collect['comment']:
                res = collect
                res["content"] = res.pop("comment")
                res["date"] = res["date"] + ' 00:00:00'
                break
        if res is not None:
            return res
        reviews = self.get_reviews(self.uid)
        for review in reviews:
            if sentence in review["content"]:
                res = review
                res["movie_id"] = res.pop("movie")
                break
        if res is not None:
            return res
        print('cannot find sentence "%s" for user %s' % (sentence, self.uid))
        return None

    def make_tags(self, profile):
        tags = []
        texts = []

        total_num = profile['info']['totalNum']
        if total_num < 50:
            tags.append('电影小白')
        elif total_num < 200:
            tags.append('电影达人')
        else:
            tags.append('阅片无数')
        texts.append(tags[-1])

        average_score = profile['distribution']['averageScore']
        if average_score > 4:
            tags.append('不吝好评')
        elif average_score > 3.5:
            tags.append('不轻易打高分')
        elif average_score > 2.5:
            tags.append('眼光挑剔')
        else:
            tags.append('找茬专家')
        texts.append(tags[-1])

        type_distribution = profile['distribution']['typeDistribution']
        texts.append('Ta最爱看这些类型的电影')
        favorite_type = sorted(type_distribution, key=lambda x: x['num'], reverse=True)[0]
        if favorite_type['num'] > 100:
            tags.append('资深%s迷' % favorite_type['type'])
        elif favorite_type['num'] > 50:
            tags.append('%s片爱好者' % favorite_type['type'])
        else:
            tags.append('刚入门%s片' % favorite_type['type'])

        worst_target = profile['sentiment']['worstTarget']
        negative_num = profile['sentiment']['negativeNum']
        negative_rate = profile['sentiment']['negativeRate']
        if round(negative_num) == 0:
            tags.append('不挑剔')
            texts.append('所有评价中均没有负面情感')
        else:
            if negative_num > 5:
                tags.append('%s吐槽狂' % worst_target)
            else:
                tags.append('挑剔%s' % worst_target)
            related_review_num = round(negative_num / negative_rate)
            if negative_rate > 0.5:
                texts.append('在涉及电影%s的%d条评价中，负面评价多达%.0f%%' % (worst_target, related_review_num, negative_rate * 100))
            else:
                texts.append('发表了%d条关于电影%s的吐槽' % (negative_num, worst_target))

        years = np.array(profile['distribution']['pubyear'])
        mid = np.median(years)
        if mid < 2000:
            tags.append('怀旧')
            texts.append('看了%d部上个世纪的电影' % np.sum(years < 2000))
        else:
            newest_num = np.sum(years > 2014)
            tags.append('紧跟潮流')
            texts.append('最近五年的新电影就看了%d部' % newest_num)

        favorite_actor = profile['favorite']['favoriteActor']
        tags.append(favorite_actor['name'])
        saw = favorite_actor['saw']
        if saw > 10:
            texts.append('可以说是铁杆粉丝了')
        else:
            texts.append('')
        profile.update({'tags': tags, 'texts': texts})

    def get_collect(self, uid, update_interval=-1, return_htmls=False):
        self.update_uid(uid)
        if cache_available(self.collect_file, update_interval):
            collects = load_json_file(self.collect_file)
            collect_htmls = load_json_file(self.collect_html_file)
        elif cache_available(self.collect_html_file, update_interval):
            collect_htmls = load_json_file(self.collect_html_file)
            collects = list(itertools.chain.from_iterable(
                map(lambda html: douban_crawler.parse_collect(None, html)[0],
                    map(lambda x: x['content'], collect_htmls))))
            save_json_file(self.collect_file, collects)
        else:
            collects, collect_htmls = douban_crawler.get_collect(uid)
            save_json_file(self.collect_file, collects)
            save_json_file(self.collect_html_file, collect_htmls)
        if return_htmls:
            return collects, collect_htmls
        collects = list(filter(lambda x: len(x) > 0, collects))
        return collects

    def get_user_info(self, uid, update_interval=-1):
        self.update_uid(uid)
        return api_crawler.get_user_info(uid, update_interval)

    def get_reviews(self, uid, update_interval=-1):
        self.update_uid(uid)
        if cache_available(self.review_file):
            reviews = load_json_file(self.review_file)
        elif cache_available(self.review_html_file, update_interval):
            review_htmls = load_json_file(self.review_html_file)
            reviews = douban_crawler.get_user_reviews(review_htmls)
            save_json_file(self.review_file, reviews)
        else:
            if cache_available(self.review_list_html_file, update_interval):
                review_list_htmls = load_json_file(self.review_list_html_file)
            else:
                review_list_htmls = None
            review_urls, htmls = douban_crawler.get_user_review_list(uid, review_list_htmls)
            if review_list_htmls is None:
                save_json_file(self.review_list_html_file, htmls)
            review_htmls = douban_crawler.get_user_review_htmls(review_urls)
            save_json_file(self.review_html_file, review_htmls)
            reviews = douban_crawler.get_user_reviews(review_htmls)
            save_json_file(self.review_file, reviews)
        reviews = list(filter(lambda x: len(x) > 0, reviews))
        return reviews


userAnalyzer = UserAnalyzer()


def is_ip_banned(s):
    html = etree.tostring(s, encoding='utf8').decode('utf8')
    return 'window.location.href="https://sec.douban.com/' in html


def craw_active_user(uid_set, collect_th=100):
    crawled_uids = []
    if os.path.exists('crawled_uid.json'):
        crawled_uids = load_json_file('crawled_uid.json')
    uid_set = uid_set - set(crawled_uids)
    print('try %d users' % (len(uid_set)))
    active_count = 0
    total_num = 0
    continue_zero_num = 0
    for uid in uid_set:
        url = 'https://www.douban.com/people/%s/' % uid
        try:
            s = etree.HTML(Limit.get_request(url, True))
        except:
            print('failed for url %s' % url)
            break
        collect_num = (s.xpath('//div[@id="movie"]/h2/span[@class="pl"]/a/text()') or ['0'])[-1]
        collect_num = int(re.search(r'\d+', collect_num).group())
        print('uid: %s, collect_num: %d' % (uid, collect_num))
        sys.stdout.flush()
        if collect_num == 0:
            continue_zero_num += 1
            if continue_zero_num >= 50 or is_ip_banned(s):
                print('check status! zero num: %d' % continue_zero_num)
                break
        else:
            continue_zero_num = 0
            crawled_uids.append(uid)
        if collect_num >= collect_th:
            userAnalyzer.get_collect(uid)
            active_count += 1
        total_num += 1
        if total_num % 50 == 0:
            print('total user: %d, active user: %d' % (total_num, active_count))
            # sys.stdout.flush()
    save_json_file('crawled_uid.json', crawled_uids)
    print('done at %s! total user: %d, crawled user: %d, active user: %d' % (
        time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime()), len(uid_set), len(crawled_uids), active_count))


def test():
    uids = set()
    movie_folder = '/data/wangke/MovieProject/MovieData/subject'
    movies = os.listdir(movie_folder)
    for i, movie in enumerate(movies):
        review_folder = os.path.join(movie_folder, movie, 'reviews')
        if not os.path.exists(review_folder):
            continue
        review_files = [os.path.join(review_folder, x) for x in os.listdir(review_folder)]
        for file in review_files:
            data = load_json_file(file)
            if 'reviews' not in data:
                continue
            uids.update([x['author']['uid'] for x in data['reviews']])
        print('got %d users, %d of %d' % (len(uids), i, len(movies)))
        sys.stdout.flush()
    craw_active_user(uids)


def re_crawl_html():
    user_list = load_np_array('/data/wangke/MovieProject/MovieData/data/user_list.npy')
    for user in user_list:
        html_file = os.path.join(CONFIG.data_path, 'user', user, 'html.json')
        if cache_available(html_file):
            logging_with_time('file exists: %s' % html_file)
            continue
        collects, htmls = douban_crawler.get_collect(user)
        save_json_file(html_file, htmls)
        logging_with_time('done: %s, html num: %d' % (user, len(htmls)))
