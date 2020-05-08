import os
import platform


class Config:
    def __init__(self):
        if platform.system().lower() == 'linux':
            self.data_path = os.path.abspath('../MovieData')
        else:
            self.data_path = 'D:\\MovieData'
        self.user_path = os.path.join(self.data_path, 'user')
        self.movie_path = os.path.join(self.data_path, 'subject')
        self.dataset_path = os.path.join(self.data_path, 'data')
        self.rate_matrix_file = os.path.join(self.dataset_path, 'rate_matrix.npy')
        self.user_list_file = os.path.join(self.dataset_path, 'user_list.npy')
        self.movie_list_file = os.path.join(self.dataset_path, 'movie_list.npy')
        self.movie_freq_file = os.path.join(self.dataset_path, 'movie_freq.txt')
        self.comment_rate_file = os.path.join(self.dataset_path, 'comment_rate.json')
        self.review_rate_file = os.path.join(self.dataset_path, 'review_rate.json')
        self.rate_record_file_name = 'rate_record'
        self.rate_record_all = os.path.join(self.dataset_path, '%s_all.txt' % self.rate_record_file_name)
        self.training_folder = os.path.join(self.dataset_path, 'train')
        self.user_profile_file = os.path.join(self.dataset_path, 'user_profile.json')
        self.movie_profile_file = os.path.join(self.dataset_path, 'movie_profile.json')
        self.user_tags_file = os.path.join(self.dataset_path, 'user_tags.json')
        self.movie_tags_file = os.path.join(self.dataset_path, 'movie_tags.json')
        self.tag_word_list = os.path.join(self.dataset_path, 'tags.txt')
        self.single_rate_comment = os.path.join(self.dataset_path, 'comment_rate.json')
        self.single_rate_review = os.path.join(self.dataset_path, 'review_rate.json')
        self.single_rate_comment_cut = os.path.join(self.dataset_path, 'comment_cut.json')
        self.single_rate_comment_profile = os.path.join(self.dataset_path, 'comment_profile.json')
        self.single_rate_review_cut = os.path.join(self.dataset_path, 'review_cut.json')
        self.sing_rate_review_profile = os.path.join(self.dataset_path, 'review_profile.json')
        self.sing_rate_data_all = os.path.join(self.dataset_path, 'single_rate_data.json')
        self.vocab_file = os.path.join(self.dataset_path, 'vocab.txt')
        self.single_rate_file_name = 'single_rate'
        self.single_rate_training_folder = os.path.join(self.dataset_path, 'single_rate_train')
        self.knowledge_folder = os.path.abspath('backend/sentiment/data')
        self.target_word_list = os.path.join(self.knowledge_folder, 'word_target.txt')
        self.description_word_list = os.path.join(self.knowledge_folder, 'word_description.txt')
        self.sentiment_category_list = os.path.join(self.knowledge_folder, 'word_sentiment.txt')
        self.upload_folder = os.path.join(self.data_path, 'upload')
        self.upload_analysis_cache_folder = os.path.join(self.upload_folder, 'cache')
        self.models_folder = os.path.join(self.data_path, 'save')
        os.makedirs(self.training_folder, exist_ok=True)
        os.makedirs(self.single_rate_training_folder, exist_ok=True)
        os.makedirs(self.upload_analysis_cache_folder, exist_ok=True)


CONFIG = Config()
