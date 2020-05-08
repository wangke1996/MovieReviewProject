import os
import multiprocessing as mp
import argparse
from backend.config import CONFIG
from backend.functionLib.function_lib import load_json_file, save_json_file, logging_with_time, read_lines, \
    load_np_array
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import gc


# shell = 'train1.py'
# thread_num = 5
# arg_names = ['num_epochs', 'i']
# arg_values = [['30'], [str(x) for x in range(17)]]


def do_task(task):
    return_code = os.system(task)
    if return_code != 0:
        logging_with_time('error %d in task: %s' % (return_code, task))
    else:
        logging_with_time('done: %s' % task)
    return return_code


def split_data(_type='comment', batch=200):
    out_folder = os.path.join(CONFIG.dataset_path, _type, 'src')
    os.makedirs(out_folder, exist_ok=True)
    in_file = CONFIG.comment_rate_file if _type == 'comment' else CONFIG.review_rate_file
    all_data = load_json_file(in_file)
    total_num = len(all_data)
    start = 0
    out_files = []
    while start < total_num:
        out_file = os.path.join(out_folder, '%d.json' % start)
        end = start + batch
        data = all_data[start:end]
        start = end
        out_files.append(out_file)
        if os.path.exists(out_file):
            continue
        save_json_file(out_file, data)
    logging_with_time('done %d files' % len(out_files))
    return out_files


def multi_pool_tasks(tasks, out_files, log_files, thread_num=10):
    tasks = list(map(lambda x, y: '%s >%s.log 2>&1' % (x, y), tasks, log_files))
    pool = mp.Pool(processes=thread_num)
    for task, out_file, log_file in zip(tasks, out_files, log_files):
        # if out_file is not None and os.path.exists(out_file):
        #     logging_with_time('skip: %s' % out_file)
        #     continue
        pool.apply_async(do_task, (task,))
    pool.close()
    pool.join()


# def test():
#     tasks = ["echo %d > thread_test_%d.txt 2>&1" % (x, x) for x in range(30)]
#     pool = mp.Pool(processes=15)
#     for i, task in enumerate(tasks):
#         pool.apply_async(do_task, (task,))
#     pool.close()
#     pool.join()


def generate_analysis_tasks(_type='comment', batch=200):
    batch_files = split_data(_type, batch)
    out_folder = os.path.join(CONFIG.dataset_path, _type, 'profile')
    cut_folder = os.path.join(CONFIG.dataset_path, _type, 'cut')
    log_folder = os.path.join(CONFIG.dataset_path, _type, 'log')
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(cut_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    tasks = []
    out_files = []
    cut_out_files = []
    log_files = []
    for in_file in batch_files:
        out_file = os.path.join(out_folder, os.path.basename(in_file))
        cut_out_file = os.path.join(cut_folder, os.path.basename(in_file))
        log_file = os.path.join(log_folder, os.path.basename(in_file))
        out_files.append(out_file)
        cut_out_files.append(cut_out_file)
        log_files.append(log_file)
        task = 'python backend/preprocess/triple_analysis.py --in_file %s --out_file %s --cut_out_file %s' % (
            in_file, out_file, cut_out_file)
        tasks.append(task)
    return tasks, out_files, log_files


def merge_results(_type='comment', batch=200):
    out_folder = os.path.join(CONFIG.dataset_path, _type, 'profile')
    cut_folder = os.path.join(CONFIG.dataset_path, _type, 'cut')
    start = 0
    out = []
    cut = []
    while True:
        out_file = os.path.join(out_folder, '%d.json' % start)
        cut_file = os.path.join(cut_folder, '%d.json' % start)
        if not os.path.exists(out_file) or not os.path.exists(cut_file):
            break
        out.extend(load_json_file(out_file))
        cut.extend(load_json_file(cut_file))
        start += batch
    out_file = os.path.join(CONFIG.dataset_path, '%s_profile_%d.json' % (_type, len(out)))
    out_cut_file = os.path.join(CONFIG.dataset_path, '%s_cut_%d.json' % (_type, len(cut)))
    save_json_file(out_file, out)
    save_json_file(out_cut_file, cut)
    print('%s saved' % out_file)


def clean_process():
    tmp_log = 'ps.tmp.log'
    os.system('ps aux |grep  backend/preprocess/triple_analysis.py > %s' % tmp_log)
    pids = read_lines(tmp_log, lambda x: x.split()[1])
    for pid in pids:
        os.system('kill %s' % pid)


def count_data(merged_cut, merged_profile):
    cut = load_json_file(merged_cut)
    sentence_num = list(map(len, cut))
    char_num = []
    for review in cut:
        num = 0
        for sentence in review:
            num += len(sentence.replace(' ', ''))
        char_num.append(num)
    total = len(cut)
    del cut
    gc.collect()
    profile = load_json_file(merged_profile)
    profile_num = list(map(len, profile))
    res = {'total': total, 'ave_char': sum(char_num) / total, 'ave_sent': sum(sentence_num) / total,
           'ave_prof': sum(profile_num) / total, 'sent': sentence_num, 'char': char_num, 'prof': profile_num}
    save_json_file('%s.count.json' % merged_profile, res)
    for k in res:
        if k.startswith('ave'):
            print('%s: %f' % (k, res[k]))
    print('total: %d' % total)


def plot_hist():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False

    comment_count = load_json_file(os.path.join(CONFIG.dataset_path, 'comment_profile.count.json'))
    review_count = load_json_file(os.path.join(CONFIG.dataset_path, 'review_profile.count.json'))

    plt.hist([comment_count['prof'], review_count['prof']], density=True, histtype='bar',
             bins=list(range(15)), color=['blue', 'red'], alpha=0.7, label=['短评', '长评'])
    plt.xlabel("观点数量")
    # plt.ylabel("")
    plt.title("观点数量分布直方图")
    # plt.show()
    plt.legend()
    plt.savefig(os.path.join(CONFIG.dataset_path, 'aa.png'))
    plt.clf()
    plt.cla()
    plt.close()


def tmp():
    user = load_np_array(CONFIG.user_list_file)
    movie = load_np_array(CONFIG.movie_list_file)
    print('user: %d, movie: %d' % (len(user), len(movie)))
    matrix = np.load(CONFIG.rate_matrix_file).astype(bool)
    user_rate_num = np.sum(matrix, axis=1)
    movie_rate_num = np.sum(matrix, axis=0)
    print('rate: %d' % sum(user_rate_num))
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.hist(user_rate_num, density=True, histtype='bar',
             bins=list(range(0, 1200, 60)), facecolor='blue', edgecolor='white', alpha=0.7)
    plt.xlabel("评论数量")
    plt.title("用户评论数量分布直方图")
    plt.savefig(os.path.join(CONFIG.dataset_path, 'bb.png'))
    plt.clf()
    plt.cla()
    plt.close()
    plt.hist(movie_rate_num, density=True, histtype='bar',
             bins=list(range(0, 30, 2)), facecolor='red', edgecolor='white', alpha=0.7)
    plt.xlabel("评论数量")
    plt.title("电影评论数量分布直方图")
    plt.savefig(os.path.join(CONFIG.dataset_path, 'cc.png'))
    plt.clf()
    plt.cla()
    plt.close()


def test():
    # plot_hist()
    tmp()


def filter_by_prof(merged_cut, merged_profile, th=3):
    count = load_json_file('%s.count.json' % merged_profile)
    filter_index = [i for i, x in enumerate(count['prof']) if x >= th]
    total = len(filter_index)
    print('filtered num: %d' % total)
    filter_lambda = lambda x: [x[i] for i in filter_index]
    char = filter_lambda(count['char'])
    sent = filter_lambda(count['sent'])
    prof = filter_lambda(count['prof'])
    new_count = {'total': total, 'ave_char': sum(char) / total, 'ave_sent': sum(sent) / total,
                 'ave_prof': sum(prof) / total, 'sent': sent, 'char': char, 'prof': prof}
    for k, v in new_count.items():
        if k.startswith('ave'):
            print('%s: %f' % (k, v))
    save_json_file('%s.count.filter' % merged_profile, new_count)

    cut = load_json_file(merged_cut)
    cut_filter = [cut[i] for i in filter_index]
    save_json_file('%s.filter' % merged_cut, cut_filter)
    del cut, cut_filter
    gc.collect()
    profile = load_json_file(merged_profile)
    profile_filter = [profile[i] for i in filter_index]
    save_json_file('%s.filter' % merged_profile, profile_filter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='analysis')
    parser.add_argument('--type', type=str, default='comment')
    parser.add_argument('--thread', type=int, default=15)
    parser.add_argument('--th', type=int, default=1)
    args = parser.parse_args()
    if args.task == 'merge':
        merge_results(args.type)
    elif args.task == 'count':
        if args.type == 'comment':
            count_data(CONFIG.single_rate_comment_cut, CONFIG.single_rate_comment_profile)
        else:
            count_data(CONFIG.single_rate_review_cut, CONFIG.sing_rate_review_profile)
    elif args.task == 'filter':
        if args.type == 'comment':
            filter_by_prof(CONFIG.single_rate_comment_cut, CONFIG.single_rate_comment_profile, args.th)
        else:
            filter_by_prof(CONFIG.single_rate_review_cut, CONFIG.sing_rate_review_profile, args.th)
    else:
        clean_process()
        cmds, outputs, logs = generate_analysis_tasks(args.type)
        multi_pool_tasks(cmds, outputs, logs, thread_num=args.thread)
    # test()


if __name__ == '__main__':
    # main()
    test()
