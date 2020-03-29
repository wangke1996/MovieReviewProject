import os
import glob
import re
from collections import Counter, defaultdict
from backend.config import CONFIG
from backend.functionLib.function_lib import load_np_array, write_lines, read_lines, logging_with_time, load_json_file, \
    save_json_file
from backend.crawler import api_crawler, Limit
from backend.analyzer.data_analyzer import dataAnalyzer
from backend.analyzer import userAnalyzer


# from backend.analyzer import dataAnalyzer


def make_movie_crawler_list(overwrite=False):
    if not overwrite and os.path.exists(CONFIG.movie_freq_file):
        movieid_freq_list = read_lines(CONFIG.movie_freq_file, lambda x: (x.split()[0], int(x.split()[1])))
        return movieid_freq_list
    movie_list = read_lines(CONFIG.rate_record_all, lambda x: int(x.split()[1]))
    sorted_movie_freq = sorted(Counter(movie_list).items(), key=lambda x: x[1], reverse=True)
    movieid_mapping = load_np_array(CONFIG.movie_list_file)
    movieid_freq_list = list(map(lambda x: (movieid_mapping[x[0]], x[1]), sorted_movie_freq))
    write_lines(CONFIG.movie_freq_file, movieid_freq_list, lambda x: '%s %d' % (x[0], x[1]))
    return movieid_freq_list


def is_movie_crawled(movie_id, crawled_movies, data_type='reviews'):
    if movie_id not in crawled_movies:
        return False
    review_num = crawled_movies[movie_id].get('reviews', 0)
    comment_num = crawled_movies[movie_id].get('comments', 0)
    if data_type == 'reviews':
        if review_num == 1000 or review_num % 100 > 0:
            return True
        else:
            return False
    else:
        if review_num > 50:
            # no need to crawl comments
            return True
        elif comment_num == 500 or comment_num % 10 > 0:
            # all comments crawled
            return True
        else:
            return False


def craw_movie_reviews(max_review_count=1000):
    # Limit.set_retry_status()
    movie_freq = make_movie_crawler_list(False)
    crawled_movies = parse_crawl_log()
    movie_freq = list(filter(lambda x: not is_movie_crawled(x[0], crawled_movies, 'reviews'), movie_freq))
    for movie, freq in movie_freq:
        try:
            review_list = api_crawler.get_movie_reviews(movie, reviews_count=max_review_count, update_interval=-1)
            review_num = len(review_list)
            assert review_num > 0, 'got zero reviews!'
            logging_with_time('movie: %s, reviews: %d, users: %d' % (movie, review_num, freq))
        except Exception as e:
            logging_with_time('error in movie %s: %s' % (movie, e))


def craw_user_reviews(user_list=None):
    if user_list is None:
        user_list = load_np_array(CONFIG.user_list_file)
    for user in user_list:
        reviews = userAnalyzer.get_reviews(user)
        logging_with_time(
            'user: %s, review num: %d, empty num: %d' % (user, len(reviews), len([x for x in reviews if len(x) == 0])))


def craw_movie_comments(max_comment_count=500):
    movie_freq = make_movie_crawler_list(False)
    crawled_movies = parse_crawl_log()
    movie_freq = list(filter(lambda x: not is_movie_crawled(x[0], crawled_movies, 'comments'), movie_freq))
    for movie, freq in movie_freq:
        try:
            comment_list = api_crawler.get_movie_comments(movie, comments_count=max_comment_count, update_interval=-1)
            comment_num = len(comment_list)
            assert comment_num > 0, 'got zero comments!'
            logging_with_time('movie: %s, comments: %d, users: %d' % (movie, comment_num, freq))
        except Exception as e:
            logging_with_time('error in movie %s: %s' % (movie, e))


def parse_crawl_log(file='crawl.done.log'):
    out_file = 'movie_review_num.json'
    # if os.path.exists(out_file):
    #     res = load_json_file(out_file)
    # else:
    res = defaultdict(dict)
    lines = read_lines(file)
    for line in lines:
        infos_review = re.search(r'movie: \d+, reviews: \d+, users: \d+', line)
        infos_comment = re.search(r'movie: \d+, comments: \d+, users: \d+', line)
        infos = infos_review or infos_comment
        if infos is None:
            continue
        infos = re.split('[:,] ', infos.group())
        movie_id = infos[1]
        data_type = infos[2]
        data_num = int(infos[3])
        res[movie_id][data_type] = max(res[movie_id].get(data_type, 0), data_num)
    save_json_file(out_file, res)
    return res


def analysis_movie_reviews():
    movie_review_num = parse_crawl_log('crawl.done.log')
    for movie in movie_review_num:
        review_num = movie_review_num[movie].get('reviews', 0)
        try:
            dataAnalyzer.analyze_movie_profile(movie, reviews_count=review_num)
        except Exception as e:
            print(e)


def analysis_movie_process():
    movie_review_num = parse_crawl_log('crawl.done.log')
    total_review = 0
    total_comment = 0
    done_review = 0
    done_comment = 0
    for movie in movie_review_num:
        review_num = movie_review_num[movie].get('reviews', 0)
        comment_num = movie_review_num[movie].get("comments", 0)
        folder = os.path.join(CONFIG.data_path, 'subject', movie, 'analysis')
        review_res_file = os.path.join(folder, 'profile_review.json')
        comment_res_file = os.path.join(folder, 'profile_comment.json')
        if os.path.exists(review_res_file):
            done_review += review_num
        if os.path.exists(comment_res_file):
            done_comment += comment_num
        total_review += review_num
        total_comment += comment_num
    print('comment: %d/%d, review: %d/%d' % (done_comment, total_comment, done_review, total_review))


def get_crawled_movie_comment_num(movie_id, _type='comments'):
    folder = os.path.join(CONFIG.movie_path, movie_id, _type)
    if not os.path.exists(folder):
        return 0
    json_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith('.json')]
    if len(json_files) == 0:
        return 0
    comments_num = [len(x[_type]) for x in map(load_json_file, json_files)]
    return sum(comments_num)


def analysis_movie_comments():
    # movie_comment_num = parse_crawl_log('crawl.done.log')
    movie_freq = make_movie_crawler_list(False)
    movie_freq.reverse()
    for movie, freq in movie_freq:
        comment_num = get_crawled_movie_comment_num(movie)
        if comment_num == 0:
            continue
        try:
            details = dataAnalyzer.analyze_movie_comments(movie, comment_num)
            if len(details) == 0:
                logging_with_time('empty result for movie: %s' % movie)
            logging_with_time('done: %s' % movie)
        except Exception as e:
            print(e)


def analysis_user_comments():
    user_list = load_np_array(CONFIG.user_list_file)
    for user in user_list:
        details = dataAnalyzer.analyze_user_comments(user)
        if len(details) == 0:
            logging_with_time('empty result for user: %s' % user)
        logging_with_time('done: %s' % user)


def analysis_user_reviews():
    user_list = load_np_array(CONFIG.user_list_file)
    for user in user_list:
        try:
            details = dataAnalyzer.analyze_user_reveiws(user)
        except OSError:
            continue
        if len(details) == 0:
            logging_with_time('empty result for user: %s' % user)
        logging_with_time('done: %s' % user)


def crawl_movie_info():
    movie_list = load_np_array(CONFIG.movie_list_file)
    count = 0
    total = len(movie_list)
    for movie in movie_list:
        api_crawler.get_movie_info(movie, -1)
        count += 1
        if count % 100 == 0:
            logging_with_time('movie info: %d/%d' % (count, total))


def prepare_user_profile():
    for i, uid in enumerate(load_np_array(CONFIG.user_list_file)):
        logging_with_time('user %d: %s' % (i, uid))
        profiles = userAnalyzer.get_basic_info(uid)
        profiles.update(userAnalyzer.get_profile_of_collect(uid))
        sentiment_profile, _ = dataAnalyzer.analyze_profile(uid, 'user')
        profiles.update(userAnalyzer.get_profile_of_sentiment(uid, sentiment_profile))
        userAnalyzer.make_tags(profiles)
