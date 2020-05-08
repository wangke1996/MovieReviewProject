import json
import os
import numpy as np
import time
import sys
import builtins


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


def print_with_time(s):
    builtins.print(time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime()) + ': ' + s)
    sys.stdout.flush()
