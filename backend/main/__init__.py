from flask import render_template
from flask import Blueprint, request
from werkzeug.utils import secure_filename
from backend.crawler.douban_api_crawler import api_crawler
from backend.analyzer import dataAnalyzer, userAnalyzer, sentimentAnalyzer
from backend.config import CONFIG
import json
import os
import numpy as np
from collections import Counter

main = Blueprint('main', __name__, template_folder='templates', static_folder='static', static_url_path="/static")


@main.route('/getMovieInTheater')
def movie_in_theater():
    return json.dumps(api_crawler.get_movie_intheaters())


@main.route('/getMovieInfo/<movie_id>')
def movie_info(movie_id):
    return json.dumps(api_crawler.get_movie_info(movie_id))


@main.route('/getMovieComments/<movie_id>/<int:count>')
def movie_comments(movie_id, count):
    return json.dumps(api_crawler.get_movie_comments(movie_id, comments_count=count))


@main.route('/getMoviePhotos/<movie_id>/<int:count>')
def movie_photos(movie_id, count):
    return json.dumps(api_crawler.get_movie_photos(movie_id, photos_count=count))


@main.route('/getMovieReviewsTrend/<movie_id>')
def movie_reviews_trend(movie_id):
    return json.dumps(dataAnalyzer.analyze_movie_reviews_trend(movie_id))


@main.route('/getTargetFreqs')
def get_target_freqs():
    _id = request.args['id']
    _type = request.args['type']
    target = request.args['target']
    return dataAnalyzer.get_target_freqs(_id, _type, target)


@main.route('/searchTarget')
def search_target():
    _id = request.args['id']
    _type = request.args['type']
    _input = request.args['input']
    return json.dumps(dataAnalyzer.search_target(_id, _type, _input))


@main.route('/getRelatedSentences')
def related_sentences():
    _id = request.args['id']
    _type = request.args['type']
    target = request.args['target']
    sentiment = request.args['sentiment']
    description = request.args['description']
    start_index = int(request.args.get('startIndex', 0))
    count = int(request.args.get('count', 3))
    return dataAnalyzer.get_related_sentences(_id, _type, target, sentiment, description, start_index, count)


@main.route('/getTargetDetail')
def target_detail():
    _id = request.args['id']
    _type = request.args['type']
    target = request.args['target']
    if target == '':
        return {}
    return dataAnalyzer.get_target_detail(_id, _type, target)


@main.route('/getTargetList')
def target_list():
    _id = request.args['id']
    _type = request.args['type']
    sort_by = request.args.get('sortBy', 'freq')
    count = request.args.get('count', 10)
    return json.dumps(dataAnalyzer.get_target_list(_id, _type, count, sort_by))


@main.route('/getUserProfile/<uid>')
def get_user_profile(uid):
    def user_profile(uid):
        if uid == 'example':
            return {
                'info': {
                    'name': '六金莱',
                    'avatar': '/source/images/avatar/SunWuKong.jpg',
                    'totalNum': 1234,
                    'homePage': 'https://baike.baidu.com/item/六小龄童/142561'
                },
                'tags': ['阅片无数', '真的很严格', '科幻迷', '剧情吐槽狂', '怀旧', '章金莱'],
                'texts': ['阅片无数', '真的很严格', '资深科幻迷', '在涉及电影剧情的40条评价中，负面评价多达80%', '看了834部上个世纪的电影', '可以说是铁杆粉丝了'],
                'distribution': {
                    'averageScore': 3,
                    'rateDistribution': [{'rate': "1 star", 'reviewNums': 264}, {'rate': "2 star", 'reviewNums': 180},
                                         {'rate': "3 star", 'reviewNums': 460}, {'rate': "4 star", 'reviewNums': 210},
                                         {'rate': "5 star", 'reviewNums': 120}],
                    'typeDistribution': [{'type': "爱情", 'num': 90}, {'type': "喜剧", 'num': 40},
                                         {'type': "动作", 'num': 120},
                                         {'type': "科幻", 'num': 426}, {'type': "纪实", 'num': 30},
                                         {'type': "艺术", 'num': 0},
                                         {'type': "恐怖", 'num': 120}, {'type': "剧情", 'num': 150},
                                         {'type': "冒险", 'num': 100},
                                         {'type': "动画", 'num': 50}, {'type': "战争", 'num': 108}],
                    'ageDistribution': [{'year': int(year), 'count': int(count)} for year, count in
                                        Counter(np.random.randint(1960, 2010, 1234, dtype=int)).items()],
                },

                'favorite': {'favoriteActor': {'id': 1274392, 'name': '章金莱', 'saw': 5,
                                               'url': 'https://movie.douban.com/celebrity/1274392/',
                                               'img': 'https://img1.doubanio.com/view/celebrity/s_ratio_celebrity/public/p1453940528.49.webp',
                                               'description': '六小龄童本名章金莱，是南猴王“六龄童”章宗义的小儿子。1959年4月12日出生于上海，祖籍浙江绍兴，现为中央电视台、中国电视剧制作中心演员剧团国家一级演员。他出生于“章氏猴戏”世家，从小随父学艺。1976年6月在上海高中毕业后，考入浙江省昆剧团艺校，专攻武生，曾主演昆剧《孙悟空三借芭蕉扇》、《美猴王大闹龙宫》、《武松打店》、《三岔口》、《挑滑车》、《战马超》等，颇受观众好评。他在央视电视剧《西游记》中扮演孙悟空，该剧在美国、日本、德国、法国及东南亚各国播出后，受到广泛好评，六小龄童从此家喻户晓、蜚声中外。'}},
                'sentiment': {
                    'reviewList': [
                        {'target': '剧情', 'description': '牵强', 'targetIndex': 33, 'descriptionIndex': 37, 'movie': '七龙珠',
                         'rate': 1, 'date': '2020-02-02 23:33:33',
                         'content': '我们不要一味跟着某一些国家后面去追他们的那种风格，什么《七龙珠》，剧情发展牵强，孙悟空都弄得髭毛乍鬼的，这个不是我们民族的东西！'},
                        {'target': '剧情', 'description': '混乱', 'targetIndex': 28, 'descriptionIndex': 30,
                         'movie': '大梦西游',
                         'rate': 1, 'date': '2019-12-23 23:33:33',
                         'content': '现在改编的这些电影，完全不尊重原著，是非颠倒，人妖不分，剧情混乱，居然还有孙悟空和白骨精谈恋爱的情节，以至于总有小朋友问我：“六爷爷，孙悟空到底有几个女朋友啊？”'},
                        {'target': '剧情', 'description': '烂', 'targetIndex': 0, 'descriptionIndex': 3, 'movie': '西游记女儿国',
                         'rate': 1, 'date': '2019-01-23 23:33:33', 'content': '剧情太烂了！戏说不是胡说，改编不是乱编，你们这样是要向全国人民谢罪的！'}, ],
                    'worstTarget': '剧情',
                    'negativeRate': 0.8,
                    'negativeNum': 32,
                }
            }
        profiles = userAnalyzer.get_basic_info(uid)
        profiles.update(userAnalyzer.get_profile_of_collect(uid))
        sentiment_profile, _ = dataAnalyzer.analyze_profile(uid, 'user')
        profiles.update(userAnalyzer.get_profile_of_sentiment(uid, sentiment_profile))
        userAnalyzer.make_tags(profiles)
        return profiles
        # return {}

    return json.dumps(user_profile(uid))


@main.route('/searchUser/<query>')
def search_user(query):
    return json.dumps(dataAnalyzer.search_user(query))


@main.route('/getActiveUsers')
def get_active_users():
    num = request.args.get('num', 10)
    return json.dumps(dataAnalyzer.get_active_users(num))


@main.route('/checkUserState/<uid>')
def check_user_state(uid):
    return dataAnalyzer.check_user_state(uid)


@main.route('/analysisUploadedFile')
def analysis_uploaded_file():
    file = request.args['file']
    if not os.path.exists(os.path.join(CONFIG.upload_folder, file)):
        return json.dumps({'cacheID': '', 'message': 'file not exists', 'status': 'error'})
    else:
        _, cache_id = sentimentAnalyzer.analysis_uploaded_file(file)
        return json.dumps({'cacheID': cache_id, 'status': 'success'})


@main.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'Error! No file uploaded'
    file = request.files['file']
    if file.filename == '':
        return 'Error! No file uploaded'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(CONFIG.upload_folder, filename))
        return filename
# @main.route('/', defaults={'path': ''})
# @main.route('/<path:path>')
# def index(path):
#     return render_template('index.html')
