from backend.crawler.douban_crawler import crawler
from backend.config import config
from backend.functionLib.function_lib import load_json_file, save_json_file, cache_available
import os

max_review_count = 10000


class DataAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def analyzeReviewsTrend(movie_id):
        folder = os.path.join(config.data_path, 'subject', movie_id, 'analysis')
        os.makedirs(folder, exist_ok=True)
        json_file = os.path.join(folder, 'reviewsTrend.json')
        if cache_available(json_file, update_interval=-1):
            results = load_json_file(json_file)
        else:
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
        # print(results)
        return results


analyzer = DataAnalyzer()
