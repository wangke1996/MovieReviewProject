import os
import re
import numpy as np
import itertools
from collections import defaultdict
from backend.config import CONFIG
from models.lib.io_helper import load_json_file, save_json_file, save_np_array, load_np_array, write_lines, read_lines, \
    print_with_time
from scipy.sparse import coo_matrix
import random
import gc


def shuffle_data_list(data_list, seed=None):
    data = list(zip(*data_list))
    if seed is None:
        random.shuffle(data)
    else:
        random.Random(seed).shuffle(data)
    return zip(*data)


def make_train_test_set(test_set_id=0, model_type=None, random_shuflle=True, random_seed=0):
    train_files = [x for x in os.listdir(CONFIG.training_folder) if
                   re.match(r'%s_\d+.txt' % CONFIG.rate_record_file_name, x) is not None]
    train_files.sort()
    test_file = '%s_%d.txt' % (CONFIG.rate_record_file_name, test_set_id)
    if test_file not in train_files:
        print('error test_set_id! No such file %s' % test_file)
        test_file = train_files[0]
        print('use %s instead' % test_file)
    train_files.remove(test_file)
    train_files = [os.path.join(CONFIG.training_folder, x) for x in train_files]
    test_file = os.path.join(CONFIG.training_folder, test_file)
    testdata = read_lines(test_file, lambda x: x.split())
    test_input = [(int(x[0]), int(x[1])) for x in testdata]
    test_labels = [int(x[2]) for x in testdata]
    traindata = []
    for train_file in train_files:
        traindata.extend(read_lines(train_file, lambda x: x.split()))
    traindata = list(map(lambda x: (int(x[0]), int(x[1]), int(x[2])), traindata))
    if random_shuflle:
        random.Random(random_seed).shuffle(traindata)
    if model_type in {'UserCF', 'ItemCF', 'LFM'}:
        dict_train_data = defaultdict(dict)
        for user, item, rate in traindata:
            dict_train_data[user][item] = rate
        traindata = dict_train_data
    return traindata, test_input, test_labels


def make_vocab_lookup(vocab_file, reverse=False, unk_token=None):
    words = read_lines(vocab_file, lambda x: x.strip().split()[0])
    if unk_token is not None and unk_token not in words:
        words.insert(0, unk_token)
    words = list(filter(lambda x: x != '', words))
    if reverse:
        # id2word
        lookup = dict([(i, x.strip()) for i, x in enumerate(words)])
    else:
        # word2id
        lookup = dict([(x.strip(), i) for i, x in enumerate(words)])
    return lookup


def make_general_vocab(vocab_size=4096, min_freq=2, overwrite=False):
    if os.path.exists(CONFIG.vocab_file) and not overwrite:
        return read_lines(CONFIG.vocab_file, lambda x: x.split()[0])
    vocab_counter = defaultdict(int)

    # def must_include_words(words):
    #     for word in words:
    #         vocab_counter[word] = 999999
    # targets = read_lines(CONFIG.target_word_list, lambda x: x.strip())
    # descriptions = read_lines(CONFIG.description_word_list, lambda x: x.strip())
    # must_include_words(targets + descriptions + ['UNK'])
    def update_with_cut_reviews(reviews):
        for review in reviews:
            for sentence in review:
                for word in sentence.split():
                    vocab_counter[word] += 1

    comments = load_json_file(CONFIG.single_rate_comment_cut)
    update_with_cut_reviews(comments)
    reviews = load_json_file(CONFIG.single_rate_review_cut)
    update_with_cut_reviews(reviews)
    items = list(filter(lambda x: x[1] >= min_freq, vocab_counter.items()))
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:vocab_size]
    write_lines(CONFIG.vocab_file, items, lambda x: '%s %d' % (x[0], x[1]))
    # write_lines(CONFIG.vocab_file, items, lambda x: x[0])


def split_dataset(k=5, overwrite=False):
    if not overwrite:
        files = [x for x in os.listdir(CONFIG.training_folder) if
                 re.match(r'%s_\d+.txt' % CONFIG.rate_record_file_name, x) is not None]
        if len(files) == k:
            return
    all_records = read_lines(CONFIG.rate_record_all)
    for i in range(k):
        out_file = os.path.join(CONFIG.training_folder, '%s_%d.txt' % (CONFIG.rate_record_file_name, i))
        subset = all_records[i::k]
        write_lines(out_file, subset, lambda x: x.strip())


def get_rate_matrix(user_th=5, mode='rate', overwrite=False):
    if not overwrite and os.path.exists(CONFIG.rate_record_all):
        try:
            user_list = load_np_array(CONFIG.user_list_file)
            movie_list = load_np_array(CONFIG.movie_list_file)
            rate_matrix = load_np_array(CONFIG.rate_matrix_file)
            return user_list, movie_list, rate_matrix
        except:
            pass
    user_index_dict = dict()
    user_list = []
    user_index = 0
    movie_index_dict = dict()
    movie_list = []
    movie_index = 0
    users = os.listdir(CONFIG.user_path)
    records = []
    for user in users:
        collect_file = os.path.join(CONFIG.user_path, user, 'collect.json')
        try:
            collects = load_json_file(collect_file)
        except:
            collects = []
        if mode == 'comment':
            collects = list(filter(lambda x: x['comment'] != '' and x["rate"] != 0, collects))
        elif mode == 'rate':
            collects = list(filter(lambda x: x["rate"] != 0, collects))
        if len(collects) < user_th:
            continue
        user_index_dict[user] = user_index
        user_list.append(user)
        user_index += 1
        # user_count[user] = len(collects)
        for item in collects:
            if mode == 'bool':
                value = 1
            else:
                value = item['rate']
            movie = item['movie_id']
            if movie not in movie_index_dict:
                movie_index_dict[movie] = movie_index
                movie_list.append(movie)
                movie_index += 1
            date = item["date"]
            row = user_index_dict[user]
            col = movie_index_dict[movie]
            # movie_count[movie] = movie_count.get(movie, 0) + 1
            records.append((row, col, value, date))
    write_lines(CONFIG.rate_record_all, records, lambda x: '%s %s %d %s' % (x[0], x[1], x[2], x[3]))
    rows, cols, values, dates = list(zip(*records))
    rate_matrix = coo_matrix((values, (rows, cols)), shape=(user_index, movie_index)).todense()
    save_np_array(CONFIG.rate_matrix_file, rate_matrix)
    save_np_array(CONFIG.user_list_file, user_list)
    save_np_array(CONFIG.movie_list_file, movie_list)
    return user_list, movie_list, rate_matrix


def make_profiles(overwrite=False):
    def parse_profile(_id, _type):
        file = os.path.join(CONFIG.data_path, _type, _id, 'analysis', 'profile.json')
        if not os.path.exists(file):
            return []
        profile = load_json_file(file)
        result = []
        for target, value in profile.items():
            for sentiment, val in value.items():
                for description, sentences in val.items():
                    result.extend([(target, description, sentiment)] * len(sentences))
        return result

    if not overwrite and os.path.exists(CONFIG.user_profile_file):
        user_profile = load_json_file(CONFIG.user_profile_file)
    else:
        users = load_np_array(CONFIG.user_list_file)
        user_profile = list(map(lambda x: parse_profile(x, 'user'), users))
        save_json_file(CONFIG.user_profile_file, user_profile)
    if not overwrite and os.path.exists(CONFIG.movie_profile_file):
        movie_profile = load_json_file(CONFIG.movie_profile_file)
    else:
        movies = load_np_array(CONFIG.movie_list_file)
        movie_profile = list(map(lambda x: parse_profile(x, 'subject'), movies))
        save_json_file(CONFIG.movie_profile_file, movie_profile)
    return user_profile, movie_profile


def make_tags(overwrite=False):
    def parse_movie_tags(movie):
        info_file = os.path.join(CONFIG.movie_path, movie, 'info.json')
        if not os.path.exists(info_file):
            return []
        info = load_json_file(info_file)
        return info.get("genres", [])

    def parse_user_tags(user):
        collect_profile_file = os.path.join(CONFIG.user_path, user, 'profile', 'collect_distribution.json')
        if not os.path.exists(collect_profile_file):
            return []
        collect_profile = load_json_file(collect_profile_file)
        tag_distribution = collect_profile.get("type", {})
        tags = list(
            itertools.chain.from_iterable([[tag for _ in range(freq)] for tag, freq in tag_distribution.items()]))
        return tags

    if not overwrite and os.path.exists(CONFIG.user_tags_file):
        user_tags = load_json_file(CONFIG.user_tags_file)
    else:
        users = load_np_array(CONFIG.user_list_file)
        user_tags = list(map(parse_user_tags, users))
        save_json_file(CONFIG.user_tags_file, user_tags)
    if not overwrite and os.path.exists(CONFIG.movie_tags_file):
        movie_tags = load_json_file(CONFIG.movie_tags_file)
    else:
        movies = load_np_array(CONFIG.movie_list_file)
        movie_tags = list(map(parse_movie_tags, movies))
        save_json_file(CONFIG.movie_tags_file, movie_tags)
    if not overwrite and os.path.exists(CONFIG.tag_word_list):
        tag_words = read_lines(CONFIG.tag_word_list, lambda x: x.strip())
    else:
        tag_words = set(itertools.chain.from_iterable(user_tags + movie_tags))
        write_lines(CONFIG.tag_word_list, tag_words)

    return user_tags, movie_tags, tag_words


def merge_cut_profile_rate(filter_no_profile=True, overwrite=False):
    if not overwrite and os.path.exists(CONFIG.sing_rate_data_all):
        return
    # cuts_comment = load_json_file(CONFIG.single_rate_comment_cut)
    # rates_comment = [x[1] for x in load_json_file(CONFIG.single_rate_comment)]
    # profiles_comment = load_json_file(CONFIG.single_rate_comment_profile)
    # data_comment = zip(cuts_comment, profiles_comment, rates_comment)
    # if filter_no_profile:
    #     data_comment = filter(lambda x: len(x[1]) > 0, data_comment)
    # data_comment = list(data_comment)
    # del cuts_comment, rates_comment, profiles_comment
    # gc.collect()
    data_comment = []
    cuts_review = load_json_file(CONFIG.single_rate_review_cut)
    rates_review = [x[1] for x in load_json_file(CONFIG.single_rate_review)]
    profiles_review = load_json_file(CONFIG.sing_rate_review_profile)
    data_review = zip(cuts_review, profiles_review, rates_review)
    if filter_no_profile:
        data_review = filter(lambda x: len(x[1]) > 0, data_review)
    data_review = list(data_review)
    del cuts_review, rates_review, profiles_review
    gc.collect()
    data = data_comment + data_review
    save_json_file(CONFIG.sing_rate_data_all, data)
    print('got %d cut_profile_rate data' % len(data))
    del data_comment, data_review, data
    gc.collect()


def split_cut_profile_rate(k=5, overwrite=False):
    if not overwrite:
        files = [x for x in os.listdir(CONFIG.single_rate_training_folder) if
                 re.match(r'%s_\d+.json' % CONFIG.single_rate_file_name, x) is not None]
        if len(files) == k:
            return
    all_records = load_json_file(CONFIG.sing_rate_data_all)
    print_with_time('all records loaded')
    random.shuffle(all_records)
    print_with_time('shuffled')
    for i in range(k):
        out_file = os.path.join(CONFIG.single_rate_training_folder,
                                '%s_%d.json' % (CONFIG.single_rate_file_name, i))
        subset = all_records[i::k]
        save_json_file(out_file, subset)
        del subset
        gc.collect()
    del all_records
    gc.collect()


def make_train_test_set_for_single_rate_pred(test_set_id=0, model_type=None, random_shuflle=False, random_seed=0):
    train_files = [x for x in os.listdir(CONFIG.single_rate_training_folder) if
                   re.match(r'%s_\d+.json' % CONFIG.single_rate_file_name, x) is not None]
    train_files.sort()
    test_file = '%s_%d.json' % (CONFIG.single_rate_file_name, test_set_id)
    if test_file not in train_files:
        print('error test_set_id! No such file %s' % test_file)
        test_file = train_files[0]
        print('use %s instead' % test_file)
    train_files.remove(test_file)
    train_files = [os.path.join(CONFIG.single_rate_training_folder, x) for x in train_files]
    test_file = os.path.join(CONFIG.single_rate_training_folder, test_file)
    testdata = load_json_file(test_file)
    test_input = [(x[0], x[1]) for x in testdata]
    test_labels = [x[2] for x in testdata]
    traindata = []
    for train_file in train_files:
        traindata.extend(load_json_file(train_file))
    if random_shuflle:
        random.Random(random_seed).shuffle(traindata)
    return traindata, test_input, test_labels


def make_rate_prediction_dataset(k=5, filter_no_profile=True, overwrite=False):
    print_with_time('merge data...')
    merge_cut_profile_rate(filter_no_profile, overwrite)
    print_with_time('make vocab ...')
    make_general_vocab(overwrite=overwrite)
    print_with_time('split dataset...')
    split_cut_profile_rate(k, overwrite)
    print_with_time('done dataset')


def get_all_comments_and_reviews():
    users = load_np_array(CONFIG.user_list_file)
    movies = load_np_array(CONFIG.movie_list_file)
    user_comment = []
    user_review = []
    movie_comment = []
    movie_review = []
    for user in users:
        try:
            review = load_json_file(os.path.join(CONFIG.user_path, user, 'review', 'review.json'))
        except Exception as e:
            print(e)
            review = []
        try:
            comment = load_json_file(os.path.join(CONFIG.user_path, user, 'collect', 'collect.json'))
        except Exception as e:
            print(e)
            comment = []
        review = [(x["content"], x["rate"]) for x in review if len(x) > 0 and x["rate"] > 0]
        comment = [(x["comment"], x["rate"]) for x in comment if x["rate"] > 0 and x["comment"].strip() != ""]
        user_review.extend(review)
        user_comment.extend(comment)
    print('user: comment: %d, review: %d' % (len(user_comment), len(user_review)))
    for movie in movies:
        review_folder = os.path.join(CONFIG.movie_path, movie, 'reviews')
        review = []
        if os.path.exists(review_folder):
            files = [os.path.join(review_folder, x) for x in os.listdir(review_folder) if x.endswith('0.json')]
            for file in files:
                data = load_json_file(file)
                review.extend(
                    [(x["content"], x["rating"]["value"]) for x in data['reviews'] if x["rating"]["value"] > 0])
        movie_review.extend(review)
        comment_folder = os.path.join(CONFIG.movie_path, movie, 'comments')
        comment = []
        if os.path.exists(comment_folder):
            files = [os.path.join(comment_folder, x) for x in os.listdir(comment_folder) if x.endswith('0.json')]
            for file in files:
                data = load_json_file(file)
                comment.extend(
                    [(x["content"], x["rating"]["value"]) for x in data['comments'] if x["rating"]["value"] > 0])
        movie_comment.extend(comment)
    print('movie: comment: %d, review: %d' % (len(movie_comment), len(movie_review)))
    comment_rate = user_comment + movie_comment
    review_rate = user_review + movie_review
    save_json_file(CONFIG.comment_rate_file, comment_rate)
    save_json_file(CONFIG.review_rate_file, review_rate)


def make_dataset(user_th=5, mode='rate', k=5, overwrite=False):
    user_list, movie_list, rate_matrix = get_rate_matrix(user_th, mode, overwrite)
    split_dataset(k, overwrite)
    make_tags(overwrite)
    make_profiles(overwrite)
    user_num = len(user_list)
    movie_num = len(movie_list)
    print('user: %d, movie: %d' % (user_num, movie_num))
    non_zero_num = sum(sum(np.array(rate_matrix).astype(bool)))
    print('element: %d, non-zero: %d' % (user_num * movie_num, non_zero_num))
