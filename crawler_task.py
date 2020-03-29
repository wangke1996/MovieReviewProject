import time
from backend.crawler.douban_api_crawler import DoubanApiCrawler

crawler: DoubanApiCrawler = DoubanApiCrawler()


def craw_intheaters_reviews():
    data = crawler.get_movie_intheaters()
    ids = [x['id'] for x in data]
    # craw_reviews(ids)
    craw_comments(ids)
    print("movies in theaters done\n\n\n")


def craw_top_reviews():
    data = crawler.get_movie_top250()
    ids = [x['id'] for x in data]
    # craw_reviews(ids)
    craw_comments(ids)
    print("movies in top250 done\n\n\n")


def craw_reviews(ids):
    clock = time.time()
    total_review_count = 0
    for id in ids:
        start_time = time.time()
        reviews = crawler.get_movie_reviews(id)
        count = len(reviews)
        total_review_count = total_review_count + count
        print('get reviews of movie %s done. items: %d, time use: %.2fs' % (id, count, time.time() - start_time))
    print('get %d movies and their reviews done, total reviews num: %d, time use: %.2f' % (
        len(ids), total_review_count, time.time() - clock))


def craw_comments(ids):
    clock = time.time()
    total_comment_count = 0
    for id in ids:
        start_time = time.time()
        comments = crawler.get_movie_comments(id)
        count = len(comments)
        total_comment_count = total_comment_count + count
        print('get comments of movie %s done. items: %d, time use: %.2fs' % (id, count, time.time() - start_time))
    print('get %d movies and their comments done, total comments num: %d, time use: %.2f' % (
        len(ids), total_comment_count, time.time() - clock))


if __name__ == "__main__":
    craw_top_reviews()
    craw_intheaters_reviews()
