import json
import time
import os


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
