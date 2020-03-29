import json
import time
import os
import re
import itertools
import jieba
import sys
import numpy as np
import glob
import hashlib
jieba.dt.cache_file = 'jieba.movie.cache'


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        json_data = json.load(f)
    return json_data


def save_json_file(file_path, json_data):
    try:
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(json_data, f, ensure_ascii=False)
    except:
        print('warning, illegal character found in file %s, use ascii instead' % (file_path))
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(json_data, f, ensure_ascii=True)


def cache_available(file, update_interval=-1):
    if not os.path.exists(file):
        return False
    m_time = os.stat(file).st_mtime
    if update_interval < 0 or (time.time() - m_time) / 3600 < update_interval:
        return True
    else:
        return False


def split_sentences(sentence):
    sentence = re.sub(r'[\u3000\n\r\xa0\ufeff]', '', sentence)
    pattern = re.compile('([﹒﹔﹖﹗,．；，。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
    sentence_list = []
    for i in pattern.split(sentence):
        if pattern.match(i) and sentence_list:
            sentence_list[-1] += i
        elif i:
            sentence_list.append(i)
    return sentence_list


def concat_list(alist):
    return list(itertools.chain.from_iterable(alist))


def cut_single_sentence(sentence):
    cut_sentence = " ".join(list(jieba.cut(sentence, cut_all=False)))
    return cut_sentence


def cut_sentences(sentence_list, thread_num=16):
    jieba.enable_parallel(thread_num)
    cut_sentence_list = ' '.join(jieba.cut('\n'.join(sentence_list))).split('\n')
    return cut_sentence_list


def search_by_char(candidate_word, input_value):
    for char in input_value:
        if char not in candidate_word:
            return False
    return True


def clean_error_crawl(folder, rule='**/*.json'):
    pattern = os.path.join(folder, rule)
    json_files = glob.glob(pattern, recursive=True)
    remove_files = []
    for file in json_files:
        data = load_json_file(os.path.join(folder, file))
        if len(data) == 0:
            remove_files.append(file)
    for file in remove_files:
        os.remove(os.path.join(folder, file))
        print('%s removed' % file)


def search_candidate(candidate_list, input_value, mode='char'):
    if mode == 'word':
        res = list(filter(lambda x: input_value in x, candidate_list))
    else:
        res = list(filter(lambda x: search_by_char(x, input_value), candidate_list))
    res.sort(key=len)
    return res


def logging_with_time(s):
    print(time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime()) + ': ' + s)
    sys.stdout.flush()


def save_np_array(file_path, array):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, array)


def load_np_array(file_path):
    return np.load(file_path)


def write_lines(file_path, array, format_function=lambda x: x):
    with open(file_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(map(format_function, array)))


def read_lines(file_path, parse_function=lambda x: x):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    return list(map(parse_function, lines))


def file_hash(file_path):
    md5 = hashlib.md5()
    buf_size = 65536
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            md5.update(data)
    return md5
