import requests
import re
import itertools
import os
from backend.functionLib.function_lib import read_lines, load_json_file, save_json_file, logging_with_time, \
    load_np_array, clean_error_crawl
from backend.config import CONFIG
from backend.crawler.douban_crawler import douban_crawler
from lxml import etree
from backend.analyzer.user_analyzer import userAnalyzer
from backend.analyzer.data_analyzer import dataAnalyzer
from collections import Counter
import numpy as np


def get_true_movie_id(movie_ids):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    movie_id_map = {}
    for movie_id in movie_ids:
        url = "https://movie.douban.com/subject/%s/" % movie_id
        r = requests.get(url, headers=headers)
        if r.url != url:
            print('redirect %s to %s' % (url, r.url))
            match = re.search(r'https://movie\.douban\.com/subject/\d+', r.url)
            if match is None:
                print('unknown redirect!')
            else:
                new_movie_id = match.group().split('/')[-1]
                assert movie_id != new_movie_id
                movie_id_map[movie_id] = new_movie_id
    return movie_id_map


def get_movie_id_to_map(log_file="test_make_date.log"):
    log_lines = read_lines(log_file)
    lost_movies = {}
    for line in log_lines:
        res = re.search(r'user: .+, \d+ movies missed in html: .+', line)
        if res is None:
            continue
        s = res.group().strip()
        user = s.split(',')[0].split(':')[-1].strip()
        movies = s.split(',')[1].split(':')[-1].strip().split()
        lost_movies[user] = movies
    return lost_movies


def make_recrawl_user(lost_movies):
    users = []
    for user, missed_movies in lost_movies.items():
        folder = os.path.join(CONFIG.user_path, user)
        collect_bk = load_json_file(os.path.join(folder, 'collect.json.bk'))
        html_file = os.path.join(CONFIG.data_path, 'user', user, 'html.json')
        htmls = [x['content'] for x in load_json_file(html_file)]
        collect = list(itertools.chain.from_iterable(map(lambda x: douban_crawler.parse_collect(None, x)[0], htmls)))
        if len(collect) < len(collect_bk):
            print('user: %s, bk: %d, now: %d' % (user, len(collect_bk), len(collect)))
            users.append(user)
    return users


def makeup_for_date():
    user_list = load_np_array('/data/wangke/MovieProject/MovieData/data/user_list.npy')
    for user in user_list:
        html_file = os.path.join(CONFIG.data_path, 'user', user, 'html.json')
        collect_file = os.path.join(CONFIG.data_path, 'user', user, 'collect.json')
        collect_file_bk = collect_file + '.bk'
        if os.path.exists(collect_file_bk):
            continue
        htmls = [x['content'] for x in load_json_file(html_file)]
        new_collects = itertools.chain.from_iterable(map(lambda x: douban_crawler.parse_collect(None, x)[0], htmls))
        old_collects = load_json_file(collect_file)
        old_collects_dict = dict(map(lambda x: (x['movie_id'], x), old_collects))
        new_collects_dict = dict(map(lambda x: (x['movie_id'], x), new_collects))
        missed_movies = set(old_collects_dict.keys()) - set(new_collects_dict.keys())
        if len(missed_movies) > 0:
            logging_with_time(
                'user: %s, %d movies missed in html: %s' % (user, len(missed_movies), ' '.join(missed_movies)))
        extra_movies = set(new_collects_dict.keys()) - set(old_collects_dict.keys())
        if len(extra_movies) > 0:
            logging_with_time(
                'user: %s, %d extra movies in html: %s' % (user, len(extra_movies), ' '.join(extra_movies)))
        for update_movie in set(old_collects_dict.keys()).intersection(set(new_collects_dict.keys())):
            old_collects_dict[update_movie].update(new_collects_dict[update_movie])

        os.rename(collect_file, collect_file_bk)
        save_json_file(collect_file, list(old_collects_dict.items()))


def continue_crawl_user_collect(user):
    folder = os.path.join(CONFIG.user_path, user)
    collect_bk_file = os.path.join(folder, 'collect.json.bk')
    collect_file = os.path.join(folder, 'collect.json')
    html_file = os.path.join(folder, 'html.json')
    os.system('cp %s %s' % (html_file, html_file + '.bk'))
    os.remove(collect_file)
    os.rename(collect_bk_file, collect_file)
    htmls = load_json_file(html_file)
    htmls = htmls[:-1]
    url = 'https://movie.douban.com' + etree.HTML(htmls[-1]["content"]).xpath('//span[@class="next"]/a/@href')[
        0].encode('utf8').decode('utf8')
    while url is not None:
        _, next_url, html = douban_crawler.parse_collect(url)
        htmls.append({'url': url, 'content': html})
        url = next_url
    save_json_file(html_file, htmls)


def remake_collect():
    user_list = load_np_array(CONFIG.user_list_file)
    for user in user_list:
        folder = os.path.join(CONFIG.data_path, 'user', user)
        remove_file = [x for x in os.listdir(folder) if x != "html.json"]
        for file in remove_file:
            os.remove(os.path.join(folder, file))
        userAnalyzer.get_collect(user)


def move():
    users = os.listdir(CONFIG.user_path)
    move_user_files = {'collect.json': 'collect/collect.json', 'html.json': 'collect/html.json',
                       'profile_comment.json': 'analysis/profile_comment.json',
                       'profile_review.json': 'analysis/profile_review.json'}
    for user in users:
        folder = os.path.join(CONFIG.user_path, user)
        for k, v in move_user_files.items():
            file = os.path.join(folder, k)
            new_file = os.path.join(folder, v)
            if not os.path.exists(file):
                continue
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            os.rename(file, new_file)
            print('%s to %s' % (file, new_file))
    movies = os.listdir(CONFIG.movie_path)
    move_movie_files = {'analysis/profile.json': 'analysis/profile_review.json'}
    for movie in movies:
        folder = os.path.join(CONFIG.movie_path, movie)
        for k, v in move_movie_files.items():
            file = os.path.join(folder, k)
            new_file = os.path.join(folder, v)
            if not os.path.exists(file):
                continue
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            os.rename(file, new_file)
            print('%s to %s' % (file, new_file))


url_403 = []


def make_up_html(url, tmp_file):
    if url in url_403:
        return ''
    print('please copy html of this url to %s: %s' % (tmp_file, url))
    input()
    with open(tmp_file, 'r', encoding='utf8') as f:
        html = f.read().strip()
    if html == '':
        url_403.append(url)
    return html


def backup_file(file):
    bk_file = file + '.bk'
    if not os.path.exists(bk_file):
        os.system('cp %s %s' % (file, bk_file))


def manual_crawl_user_review_with_login():
    user_list = load_np_array(CONFIG.user_list_file)
    tmp_file = 'tmp.html'
    for user in user_list:
        userAnalyzer.update_uid(user)
        backup_file(userAnalyzer.review_file)
        backup_file(userAnalyzer.review_html_file)
        backup_file(userAnalyzer.review_list_html_file)
        if not os.path.exists(userAnalyzer.review_list_html_file):
            print('review list html missed: %s' % user)
            continue
        if not os.path.exists(userAnalyzer.review_html_file):
            print('review html missed: %s' % userAnalyzer)
            continue
        # if some html content of review_list_html is empty
        review_list_htmls = load_json_file(userAnalyzer.review_list_html_file)
        review_list_changed = []
        for i, html in enumerate(review_list_htmls):
            if html["content"] == "":
                new_content = make_up_html(html["url"], tmp_file)
                html["content"] = new_content
                if new_content != "":
                    review_list_changed.append(i)
        review_htmls = load_json_file(userAnalyzer.review_html_file)
        if len(review_list_changed) > 0:
            # update review_htmls
            save_json_file(userAnalyzer.review_list_html_file, review_list_htmls)
            for new_review_list_htmls in [review_list_htmls[i] for i in review_list_changed]:
                new_urls, _, = douban_crawler.get_user_review_list(user, new_review_list_htmls, False)
                new_review_htmls = douban_crawler.get_user_review_htmls(new_urls)
                review_htmls.extend(new_review_htmls)
        save_json_file(userAnalyzer.review_html_file, review_htmls)

        s = etree.HTML(review_list_htmls[0]["content"])
        title = (s.xpath('//div[@id="db-usr-profile"]/div[@class="info"]/h1/text()') or [''])[0]
        if title == '':
            print('Error in review list page of %s, check this page and maybe your cache html' % user)
            review_num = 0
        else:
            review_num = int(title.split('(')[-1].split(')')[0])
        review_urls = [x["url"] for x in review_htmls]
        # if review_urls not all parsed
        review_html_changed = False
        if review_num != len(review_urls):
            print('unmatched review num: expected %d, got %d' % (review_num, len(review_urls)))
            print("recrawl review_list_htmls for user %s" % user)
            os.remove(userAnalyzer.review_list_html_file)
            new_review_urls, review_list_htmls = douban_crawler.get_user_review_list(user)
            save_json_file(userAnalyzer.review_list_html_file, review_list_htmls)
            added_review_urls = list(filter(lambda x: x not in review_urls, new_review_urls))
            print("to crawl %d new reviews" % (len(added_review_urls)))
            new_review_htmls = douban_crawler.get_user_review_htmls(added_review_urls)
            review_htmls.extend(new_review_htmls)
            save_json_file(userAnalyzer.review_html_file, review_htmls)
            print("done")
            review_html_changed = True
        # if html content of review html is empty
        for html in review_htmls:
            url = html["url"]
            content = html["content"]
            if content == "":
                new_content = make_up_html(url, tmp_file)
                html["content"] = new_content
                review_html_changed = True
        if review_html_changed:
            save_json_file(userAnalyzer.review_html_file, review_htmls)
            new_reviews = douban_crawler.get_user_reviews(review_htmls)
            save_json_file(userAnalyzer.review_file, new_reviews)
    print(url_403)


def analysis_process():
    comment_res_file = 'analysis/profile_comment.json'
    review_res_file = 'analysis/profile_review.json'

    users = load_np_array(CONFIG.user_list_file)
    done_comment = [os.path.exists(os.path.join(CONFIG.user_path, user, comment_res_file)) for user in users]
    done_review = [os.path.exists(os.path.join(CONFIG.user_path, user, review_res_file)) for user in users]
    print("user comment: %d/%d" % (sum(done_comment), len(users)))
    print("user review: %d/%d" % (sum(done_review), len(users)))

    movies = load_np_array(CONFIG.movie_list_file)
    done_comment = [os.path.exists(os.path.join(CONFIG.movie_path, movie, comment_res_file)) for movie in movies]
    done_review = [os.path.exists(os.path.join(CONFIG.movie_path, movie, review_res_file)) for movie in movies]
    print("movie comment: %d/%d" % (sum(done_comment), len(movies)))
    print("movie review: %d/%d" % (sum(done_review), len(movies)))


def merge_profile():
    users = load_np_array(CONFIG.user_list_file)
    nums = []
    for user in users:
        proifle = dataAnalyzer.merge_review_comment_profiles(user, 'user')
        triple_num = sum([sum([len(y.values()) for y in x.values()]) for x in proifle.values()])
        nums.append(triple_num)
    print('user: %d' % len(users))
    print('mean: %f, median: %d, max: %d' % (np.mean(nums), np.median(nums), np.max(nums)))
    print(Counter(nums))
    movies = load_np_array(CONFIG.movie_list_file)
    nums = []
    for movie in movies:
        proifle = dataAnalyzer.merge_review_comment_profiles(movie)
        triple_num = sum([sum([len(y.values()) for y in x.values()]) for x in proifle.values()])
        nums.append(triple_num)
    print('movie: %d' % len(movies))
    print('mean: %f, median: %d, max: %d' % (np.mean(nums), np.median(nums), np.max(nums)))
    print(Counter(nums))


def main():
    # remake_collect()
    # move()
    # manual_crawl_user_review_with_login()
    # analysis_process()
    # merge_profile()
    clean_error_crawl(CONFIG.movie_path, '*/info.json')
