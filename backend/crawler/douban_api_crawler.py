from .limit import Limit
from backend.config import CONFIG
from backend.functionLib.function_lib import load_json_file, save_json_file, cache_available, logging_with_time
import os
import json
import time
from random import choice


class DoubanApiCrawler:
    def __init__(self):
        # self.agents = ['douban.uieee.com', 'douban-api.uieee.com', 'douban-api.now.sh', 'douban-api.zce.now.sh',
        #                'douban-api-git-master.zce.now.sh', 'api.douban.com']
        # self.error_agents = ['douban-api-git-master.zce.now.sh']
        self.error_agents = []
        self.agents = ['api.douban.com']
        self.key_needed_agents = ['api.douban.com']
        self.key = ['0b2bdeda43b5688921839c8ecb20399b', '0df993c66c0c636e29ecbb5344252a4a']

        self.data_path = CONFIG.data_path

    def make_request_url(self, route_path, url_type='movie', parameters: dict = None):
        agent = choice(list(set(self.agents) - set(self.error_agents)))
        url = 'https://%s/v2/%s/%s' % (agent, url_type, route_path)
        if parameters is None:
            parameters = {}
        if agent in self.key_needed_agents:
            parameters['apikey'] = choice(self.key)
        if len(parameters) > 0:
            url += '?' + '&'.join(map(lambda x: '%s=%s' % (str(x[0]), str(x[1])), parameters.items()))
        return url

    @staticmethod
    def get_json_data(json_file, url, update_interval=-1):
        if cache_available(json_file, update_interval):
            # 本地缓存可用且未过期，则直接读取数据
            json_data = load_json_file(json_file)
        else:
            # 加载本地缓存失败，爬取并写入缓存
            try:
                json_data = json.loads(Limit.get_request(url), encoding='utf8')
            except:
                json_data = {}
            # 检查错误信息
            if len(json_data) == 0 or 'msg' in json_data or 'code' in json_data or 'request' in json_data:
                print('Error in crawling file: %s, url: %s' % (json_file, url))
                json_data = {}
            save_json_file(json_file, json_data)
        return json_data

    def get_object_item(self, route_path, cache_file_name, update_interval=-1, url_type='movie', data_path=None):
        """
        通过豆瓣API爬取某个特定的对象，可用于爬取电影信息、演员信息等
        :param route_path: self.ulr_pre+'/'+route_path=调用的API的网址
        :param cache_file_name: 本地缓存json文件的文件名
        :param update_interval: 本地缓存的生命周期（小时），-1代表永久有效
        :return: 爬取到的json object
        """
        if data_path is not None:
            folder = os.path.join(self.data_path, data_path)
        else:
            folder = os.path.join(self.data_path, route_path)
        os.makedirs(folder, exist_ok=True)
        json_file = os.path.join(folder, '%s.json' % cache_file_name)
        url = self.make_request_url(route_path, url_type)
        return self.get_json_data(json_file, url, update_interval=update_interval)

    def get_list_items(self, route_path, field_name, require_count=-1, update_interval=-1, url_type='movie',
                       max_retry=3):
        """
        通过豆瓣API爬取某个特定的列表，可用于爬取评论、电影榜单等
        :param route_path: self.ulr_pre+'/'+route_path=调用的API的网址
        :param field_name: 返回值中保留的字段
        :param require_count: 需要爬取的item数量
        :param update_interval: 本地缓存的生命周期（小时），-1代表永久有效
        :return: 抓取到的item list
        """
        folder = os.path.join(self.data_path, route_path)
        os.makedirs(folder, exist_ok=True)
        start = 0
        count = 100  # 本地缓存每个json文件包含的评论数量，最大为100
        items = []
        while require_count < 0 or start < require_count:
            json_file = os.path.join(folder, '%d.json' % start)
            retry = 0
            new_data = []
            while True and retry <= max_retry:
                url = self.make_request_url(route_path, url_type, {'start': start, 'count': count})
                json_data = self.get_json_data(json_file, url, update_interval=update_interval)
                if field_name not in json_data:
                    break
                new_data = json_data[field_name]
                if len(new_data) == 0 and json_data['count'] > 0:
                    logging_with_time('need rest! url: %s' % url)
                    os.remove(json_file)
                    time.sleep(20)
                    retry += 1
                else:
                    break
            items.extend(new_data)
            start = start + count
            if len(new_data) == 0 or 0 < require_count <= start or start >= json_data['total']:
                break  # 已爬取所需数量的信息或已达可获取信息上限
        if len(items) > require_count >= 0:
            items = items[:require_count]
        return items

    def get_movie_info(self, movie_id, update_interval=-1):
        """
        爬取电影信息
        :param movie_id: 要爬取的电影id
        :param update_interval: 本地缓存的生命周期（小时），-1代表缓存永久有效
        :return: movie info
        """
        route_path = 'subject/%s' % movie_id
        return self.get_object_item(route_path, 'info', update_interval)

    def get_movie_reviews(self, movie_id, reviews_count=-1, update_interval=-1):
        """
        爬取电影长评
        :param movie_id: movie id
        :param reviews_count: 需要获取的评论条数,-1代表全部爬取
        :param update_interval: 本地缓存的生命周期（小时），-1代表永久有效
        :return: 长评list
        """
        route_path = 'subject/%s/reviews' % movie_id
        return self.get_list_items(route_path, 'reviews', require_count=reviews_count, update_interval=update_interval)

    def get_movie_comments(self, movie_id, comments_count=-1, update_interval=-1):
        """
        爬取电影短评
        :param movie_id: movie id
        :param comments_count: 需要获取的短评条数,-1代表全部爬取
        :param update_interval: 本地缓存的生命周期（小时），-1代表永久有效
        :return: 短评list
        """
        route_path = 'subject/%s/comments' % movie_id
        return self.get_list_items(route_path, 'comments', require_count=comments_count,
                                   update_interval=update_interval)

    def get_movie_photos(self, movie_id, photos_count=-1, update_interval=-1):
        """
        爬取电影剧照
        :param movie_id: movie id
        :param photos_count: 爬取剧照数量
        :param update_interval: 本地缓存生命周期（小时），-1代表永久有效
        :return: photos list
        """
        route_path = 'subject/%s/photos' % movie_id
        return self.get_list_items(route_path, 'photos', require_count=photos_count, update_interval=update_interval)

    def get_movie_intheaters(self, movie_count=-1, update_interval=24):
        """
        爬取正在上映的电影
        :param movie_count: 爬取电影数量
        :param update_interval: 本地缓存生命周期（小时），-1代表永久有效
        :return: movie list
        """
        route_path = 'in_theaters'
        return self.get_list_items(route_path, 'subjects', require_count=movie_count, update_interval=update_interval)

    def get_movie_top250(self, movie_count=250, update_interval=-1):
        """
        爬取最受欢迎的电影
        :param movie_count: 爬取电影数量
        :param update_interval: 本地缓存的生命周期（小时），-1代表永久有效
        :return: movie list
        """
        route_path = 'top250'
        return self.get_list_items(route_path, 'subjects', require_count=movie_count, update_interval=update_interval)

    def get_celebrity_info(self, celebrity_id, update_interval=-1):
        """
        爬取影人信息
        :param celebrity_id: 影人id
        :param update_interval: 本地缓存有效期（小时）
        :return: 影人info
        """
        route_path = 'celebrity/%s' % celebrity_id
        return self.get_object_item(route_path, 'info', update_interval)

    def get_user_info(self, uid: str, update_interval=-1):
        """
        爬取用户信息
        :param uid: 用户id
        :param update_interval: 本地缓存有效期（小时）
        :return: 用户info
        """
        data_path = 'user/%s' % uid
        return self.get_object_item(uid, 'info', update_interval, 'user', data_path)

    def get_celebrity_works(self, celebrity_id, works_count=-1, update_interval=720):
        """
        爬取影人作品
        :param celebrity_id: 影人id
        :param update_interval: 本地缓存有效期（小时）
        :param works_count: 爬取影人作品数量
        :return: 影人作品list
        """
        route_path = 'celebrity/%s/works' % celebrity_id
        return self.get_list_items(route_path, 'works', require_count=works_count, update_interval=update_interval)

    def get_celebrity_photos(self, celebrity_id, photos_count=-1, update_interval=5040):
        """
        爬取影人照片
        :param celebrity_id: 影人id
        :param update_interval: 本地缓存有效期（小时）
        :param photos_count: 爬取影人照片数量
        :return: 影人照片list
        """
        route_path = 'celebrity/%s/photos' % celebrity_id
        return self.get_list_items(route_path, 'photos', require_count=photos_count, update_interval=update_interval)


api_crawler = DoubanApiCrawler()


def crawler_test():
    crawler: DoubanApiCrawler = DoubanApiCrawler()
    test_movie_id = "26266893"
    test_celebrity_id = "1000525"

    start_time = time.time()
    data = crawler.get_movie_intheaters()
    print('get_movie_intheaters done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))

    start_time = time.time()
    data = crawler.get_movie_top250()
    print('get_movie_top250 done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))

    start_time = time.time()
    data = crawler.get_celebrity_info(test_celebrity_id)
    print('get_celebrity_info done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))

    start_time = time.time()
    data = crawler.get_celebrity_photos(test_celebrity_id)
    print('get_celebrity_photos done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))

    start_time = time.time()
    data = crawler.get_celebrity_works(test_celebrity_id)
    print('get_celebrity_works done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))

    start_time = time.time()
    data = crawler.get_movie_info(test_movie_id)
    print('get_movie_info done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))

    start_time = time.time()
    data = crawler.get_movie_reviews(test_movie_id)
    print('get_movie_reviews done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))

    start_time = time.time()
    data = crawler.get_movie_comments(test_movie_id)
    print('get_movie_comments done. items: %d, time use: %.2fs' % (len(data), time.time() - start_time))
