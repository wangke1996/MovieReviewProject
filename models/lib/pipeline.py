import sys
import os
from models.lib.data_helper import make_dataset, make_train_test_set, make_rate_prediction_dataset, \
    make_train_test_set_for_single_rate_pred
from models.lib.scorer import calculate_score
from models.lib.io_helper import print_with_time
from collections import defaultdict
from backend.config import CONFIG

print = print_with_time
stdout = sys.stdout
stderr = sys.stderr


class Pipeline:
    def __init__(self, model_class, model_type=None, k_fold=5, remake_dataset=False,
                 metrics=('ACC', 'MSE', 'macro_F1'), task='multi'):
        if model_type is None:
            model_type = model_class.__name__
        self.model_type = model_type
        log_folder = os.path.join(CONFIG.models_folder, 'logs')
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, model_type + '.log')
        print('log file: %s' % log_file)
        self.file = open(log_file, 'w+', encoding='utf8')
        sys.stdout = self.file
        sys.stderr = self.file
        self.k_fold = k_fold
        self.model_class = model_class
        self.model = None
        self.model_dir = None
        self.trainset = None
        self.test_input = None
        self.test_label = None
        self.metrics = metrics
        self.result = defaultdict(list)
        self.test_set_id = None
        self.task = task
        print('read user-item-rate data')
        if task == 'single':
            make_rate_prediction_dataset(k=k_fold, overwrite=remake_dataset)
        else:
            make_dataset(k=k_fold, overwrite=remake_dataset)

    def init_model(self, **kwargs):
        model_dir = '%s-%d' % (self.model_type, self.test_set_id)
        if 'pretrain_dirs' in kwargs:
            pretrain_dirs = tuple(
                [os.path.join(CONFIG.models_folder, 'model/%s-%d/ckpt') % (d, self.test_set_id) for d in
                 kwargs.get('pretrain_dirs')])
            kwargs['pretrain_dirs'] = pretrain_dirs
        self.model_dir = model_dir
        print('initial model and dataset for %s' % model_dir)
        self.model = self.model_class(model_dir=model_dir, **kwargs)
        make_dataset_fun = make_train_test_set_for_single_rate_pred if self.task == 'single' else make_train_test_set
        self.trainset, self.test_input, self.test_label = make_dataset_fun(test_set_id=self.test_set_id,
                                                                           model_type=self.model_class.__name__)
        if 'make_dataset' in dir(self.model_class):
            self.model.make_dataset(self.trainset, self.test_input, self.test_label)
        print('initial done')

    def train(self):
        print('training for %s' % self.model_dir)
        self.model.fit(trainset=self.trainset)
        print('training done')

    def test(self):
        if 'from_best_model' in self.model.prediction.__code__.co_varnames:
            predictions = self.model.prediction(self.test_input, from_bets_model=True)
        else:
            predictions = self.model.prediction(self.test_input)
        self.model.model_manager.save_pkl(predictions, 'predictions')
        for metric in self.metrics:
            res = calculate_score(predictions, self.test_label, metric, 1, 5)
            print('test_set: %d, %s: %f' % (self.test_set_id, metric, res))
            self.result[metric].append(res)

    def run(self, **kwargs):
        for k in range(self.k_fold):
            self.test_set_id = k
            self.init_model(**kwargs)
            self.train()
            self.test()
        print('done')
        for metric, res in self.result.items():
            average_value = sum(res) / len(res)
            max_value = max(res)
            print('average %s: %f ± %f' % (metric, average_value, max_value - average_value))
        self.file.close()
        if 'clean' in dir(self.model_class):
            self.model.clean()
        sys.stdout = stdout
        sys.stderr = stderr
