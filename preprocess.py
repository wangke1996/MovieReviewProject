import argparse
from backend.sentiment import *
from backend.config import CONFIG
from backend.functionLib.function_lib import clean_error_crawl
from backend.preprocess.craw_movie_reviews import craw_movie_reviews, analysis_movie_reviews, craw_movie_comments, \
    analysis_user_comments, analysis_movie_comments, craw_user_reviews, analysis_user_reviews, crawl_movie_info, \
    prepare_user_profile


def test():
    # craw_user_reviews(['perhapsfish'])
    prepare_user_profile()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='comment', help='review | comment | analyze | user_comment')
    args = parser.parse_args()
    # clean_error_crawl(CONFIG.movie_path)
    if args.task == 'review':
        craw_movie_reviews(max_review_count=1000)
    elif args.task == 'comment':
        craw_movie_comments(max_comment_count=500)
    elif args.task == 'craw_review_user':
        craw_user_reviews()
    elif args.task == 'user_comment':
        analysis_user_comments()
    elif args.task == "user_review":
        analysis_user_reviews()
    elif args.task == 'movie_comment':
        analysis_movie_comments()
    elif args.task == 'movie_info':
        crawl_movie_info()
    elif args.task == 'test':
        test()
    else:
        analysis_movie_reviews()
