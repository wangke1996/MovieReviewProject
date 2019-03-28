from flask import render_template
from flask import Blueprint
from backend.crawler.douban_crawler import crawler
from backend.analyzer.data_analyzer import analyzer
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
    print('get movie photos: %s' % movie_id)
    return json.dumps(crawler.get_movie_photos(movie_id, photos_count=count))


@main.route('/getMovieReviewsTrend/<movie_id>')
def movie_reviews_trend(movie_id):
    print('get review trend: %s' % movie_id)
    return json.dumps(analyzer.analyzeReviewsTrend(movie_id))


@main.route('/', defaults={'path': ''})
@main.route('/<path:path>')
def index(path):
    return render_template('index.html')
