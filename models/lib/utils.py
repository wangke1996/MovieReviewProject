import time
import pickle

import os
import shutil
from models.lib.io_helper import save_json_file, load_json_file
from backend.config import CONFIG


class LogTime:
    """
    Time used help.
    You can use count_time() in for-loop to count how many times have looped.
    Call finish() when your for-loop work finish.
    WARNING: Consider in multi-for-loop, call count_time() too many times will slow the speed down.
            So, use count_time() in the most outer for-loop are preferred.
    """

    def __init__(self, print_step=20000, words=''):
        """
        How many steps to print a progress log.
        :param print_step: steps to print a progress log.
        :param words: help massage
        """
        self.proccess_count = 0
        self.PRINT_STEP = print_step
        # record the calculate time has spent.
        self.start_time = time.time()
        self.words = words
        self.total_time = 0.0

    def count_time(self):
        """
        Called in for-loop.
        :return:
        """
        # log steps and times.
        if self.proccess_count % self.PRINT_STEP == 0:
            curr_time = time.time()
            print(self.words + ' steps(%d), %.2f seconds have spent..' % (
                self.proccess_count, curr_time - self.start_time))
        self.proccess_count += 1

    def finish(self):
        """
        Work finished! Congratulations!
        :return:
        """
        print('total %s step number is %d' % (self.words, self.get_curr_step()))
        print('total %.2f seconds have spent\n' % self.get_total_time())

    def get_curr_step(self):
        return self.proccess_count

    def get_total_time(self):
        return time.time() - self.start_time


class ModelManager:
    """
    Model manager is designed to load and save all models.
    No matter what dataset name.
    """

    def __init__(self, name):
        self.path_name = os.path.join(CONFIG.models_folder, "model", name)
        os.makedirs(self.path_name, exist_ok=True)

    def remove_dir(self, dir_name, not_found_warning=False):
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        elif os.path.exists(os.path.join(self.path_name, dir_name)):
            shutil.rmtree(os.path.join(self.path_name, dir_name))
        elif not_found_warning:
            print('cannot remove non-exists dir %s' % dir_name)

    def make_dir_for_file(self, file_path):
        dir_name = os.path.dirname(os.path.join(self.path_name, file_path))
        os.makedirs(dir_name, exist_ok=True)

    def save_pkl(self, data, save_name: str):
        """
        Save model to model/ dir.
        :param data: source model
        :param save_name: model saved name.
        :return: None
        """
        if 'pkl' not in save_name:
            save_name += '.pkl'
        self.make_dir_for_file(save_name)
        pickle.dump(data, open(os.path.join(self.path_name, save_name), "wb"))

    def save_json(self, data, save_name: str):
        if '.json' not in save_name:
            save_name += '.json'
        self.make_dir_for_file(save_name)
        save_json_file(os.path.join(self.path_name, save_name), data)

    def load_pkl(self, save_name: str):
        """
        Load model from model/ dir via model name.
        :param save_name:
        :return: loaded model
        """
        if 'pkl' not in save_name:
            save_name += '.pkl'
        file = os.path.join(self.path_name, save_name)
        if not os.path.exists(file):
            raise OSError('There is no such model: %s' % file)
        return pickle.load(open(file, "rb"))

    def load_json(self, save_name: str):
        if '.json' not in save_name:
            save_name += '.json'
        file = os.path.join(self.path_name, save_name)
        if not os.path.exists(file):
            raise OSError('No such json file: %s' % file)
        return load_json_file(file)
