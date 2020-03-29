import os


class Config:
    def __init__(self):
        self.data_path = os.path.abspath('../MovieData')
        self.user_path = os.path.join(self.data_path, 'user')
        self.movie_path = os.path.join(self.data_path, 'subject')
        self.dataset_path = os.path.join(self.data_path, 'data')
        self.rate_matrix_file = os.path.join(self.dataset_path, 'rate_matrix.npy')
        self.user_list_file = os.path.join(self.dataset_path, 'user_list.npy')
        self.movie_list_file = os.path.join(self.dataset_path, 'movie_list.npy')
        self.movie_freq_file = os.path.join(self.dataset_path, 'movie_freq.txt')
        self.rate_record_file_name = 'rate_record'
        self.rate_record_all = os.path.join(self.dataset_path, '%s_all.txt' % self.rate_record_file_name)
        self.training_folder = os.path.join(self.dataset_path, 'train')
        os.makedirs(self.training_folder, exist_ok=True)
        self.upload_folder = os.path.join(self.data_path, 'upload')
        self.upload_analysis_cache_folder = os.path.join(self.upload_folder, 'cache')
        os.makedirs(self.upload_analysis_cache_folder, exist_ok=True)


CONFIG = Config()
