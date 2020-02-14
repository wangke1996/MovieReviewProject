from flask import render_template
from flask import Blueprint, request
from backend.crawler.douban_crawler import crawler
from backend.analyzer import dataAnalyzer
import json

main = Blueprint('main', __name__, template_folder='templates', static_folder='static', static_url_path="/static")


@main.route('/getMovieInTheater')
def movie_in_theater():
    return json.dumps(crawler.get_movie_intheaters())


@main.route('/getMovieInfo/<movie_id>')
def movie_info(movie_id):
    return json.dumps(crawler.get_movie_info(movie_id))


@main.route('/getMovieComments/<movie_id>/<int:count>')
def movie_comments(movie_id, count):
    return json.dumps(crawler.get_movie_comments(movie_id, comments_count=count))


@main.route('/getMoviePhotos/<movie_id>/<int:count>')
def movie_photos(movie_id, count):
    return json.dumps(crawler.get_movie_photos(movie_id, photos_count=count))


@main.route('/getMovieReviewsTrend/<movie_id>')
def movie_reviews_trend(movie_id):
    return json.dumps(dataAnalyzer.analyzeReviewsTrend(movie_id))


@main.route('/getTargetFreqs/<movie_id>/<target>')
def get_target_freqs(movie_id, target):
    return dataAnalyzer.get_target_freqs(movie_id, target)


@main.route('/searchTarget/<movie_id>/<input_value>')
def search_target(movie_id, input_value):
    return json.dumps(dataAnalyzer.search_target(movie_id, input_value))


@main.route('/getRelatedSentences')
def related_sentences():
    movie_id = request.args['movieID']
    target = request.args['target']
    sentiment = request.args['sentiment']
    description = request.args['description']
    start_index = int(request.args.get('startIndex', 0))
    count = int(request.args.get('count', 3))
    return dataAnalyzer.get_related_sentences(movie_id, target, sentiment, description, start_index, count)


@main.route('/getTargetDetail')
def target_detail():
    movie_id = request.args['movieID']
    target = request.args['target']
    if target == '':
        return {}
    return dataAnalyzer.get_target_detail(movie_id, target)


@main.route('/getTargetList')
def target_list():
    movie_id = request.args['movieID']
    sort_by = request.args.get('sortBy', 'freq')
    count = request.args.get('count', 10)
    return json.dumps(dataAnalyzer.get_target_list(movie_id, count, sort_by))
# @main.route('/', defaults={'path': ''})
# @main.route('/<path:path>')
# def index(path):
#     return render_template('index.html')
