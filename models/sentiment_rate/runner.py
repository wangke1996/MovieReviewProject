import os
import numpy as np
import tensorflow as tf
from models.lib import utils, CONFIG
from models.lib.data_helper import load_np_array, make_vocab_lookup, load_json_file, shuffle_data_list
from models.lib.io_helper import print_with_time
from .triple_representation import TripleSentimentRating
from random import shuffle
from sklearn.metrics import accuracy_score
from collections import Counter


class SentimentRating(object):
    def __init__(self, model_dir, batch_size=8192, epochs=100, lr=1e-3, dropout=0.5, early_stop=10, max_length=50,
                 overwrite=True, **kwargs):
        self.model_manager = utils.ModelManager(model_dir)
        self.ckpt_path = os.path.join(self.model_manager.path_name, 'ckpt')
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.early_stop = early_stop
        self.max_length = max_length
        self.user_list = load_np_array(CONFIG.user_list_file)
        self.item_list = load_np_array(CONFIG.movie_list_file)
        self.target_word2id = make_vocab_lookup(CONFIG.target_word_list, unk_token='UNK')
        self.description_word2id = make_vocab_lookup(CONFIG.description_word_list, unk_token='UNK')
        self.sentiment_word2id = make_vocab_lookup(CONFIG.sentiment_category_list)
        self.target_num = len(self.target_word2id)
        self.description_num = len(self.description_word2id)

        tf.reset_default_graph()
        self.model = TripleSentimentRating(self.target_num, self.description_num, **kwargs)
        self.sess = None
        self.saver = None
        self.global_step = tf.Variable(0, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope("Optimizer"):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.model.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            optimizer = tf.train.AdamOptimizer(self.lr)
            # clipped_gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            # optimizer = tf.train.GradientDescentOptimizer(self.lr)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        self.create_or_load_model(overwrite)

        def parse_profile(_id, _type='subject'):
            file = os.path.join(CONFIG.data_path, _type, str(_id), 'analysis', 'profile.json')
            targets = []
            descriptions = []
            sentiments = []
            freqs = []
            if not os.path.exists(file):
                print_with_time('file not exists: %s' % file)
                return {'target': targets, 'description': descriptions, 'sentiment': sentiments, 'freq': freqs,
                        'length': 0}
            profile = load_json_file(file)
            for target, sentiment_description_sample in profile.items():
                for sentiment, description_sample in sentiment_description_sample.items():
                    for description, samples in description_sample.items():
                        targets.append(target)
                        descriptions.append(description)
                        sentiments.append(sentiment)
                        freqs.append(len(samples))
            targets = list(map(lambda x: self.target_word2id[x], targets))
            descriptions = list(map(lambda x: self.description_word2id[x], descriptions))
            sentiments = list(map(lambda x: self.sentiment_word2id[x], sentiments))
            length = len(freqs)
            return {'target': targets, 'description': descriptions, 'sentiment': sentiments, 'freq': freqs,
                    'length': length}

        print_with_time('initial user profiles')
        try:
            self.user_profiles = self.model_manager.load_json('user_profiles')
        except OSError:
            self.user_profiles = list(map(lambda x: parse_profile(x, 'user'), self.user_list))
            self.model_manager.save_json(self.user_profiles, 'user_profiles')
        print_with_time('initial movie profiles')
        try:
            self.movie_profiles = self.model_manager.load_json('movie_profiles')
        except OSError:
            self.movie_profiles = list(map(lambda x: parse_profile(x, 'subject'), self.item_list))
            self.model_manager.save_json(self.movie_profiles, 'movie_profiles')
        print_with_time('profiles initialized')

    def get_profile(self, _id, profile_list):
        profile = profile_list[_id]
        if self.max_length is not None and profile['length'] > self.max_length:
            # shuffle then trunc by self.max_length
            items = [(k, v) for k, v in profile.items() if k != 'length']
            keys, vals = zip(*items)
            shuffled_vals = shuffle_data_list(vals)
            trunced_vals = list(zip(*shuffled_vals))[:self.max_length]
            new_profile = dict(zip(keys, zip(*trunced_vals)))
            new_profile['length'] = self.max_length
            return new_profile
        else:
            return profile

    def batch_iter(self, users, items, rates, shuffle_every_epoch=True, epoch_num=1):
        def parse_batch_profile(ids, profile_list):
            profiles = list(map(lambda x: self.get_profile(x, profile_list), ids))
            length = [x['length'] for x in profiles]
            max_length = max(length)
            batch_profiles = dict()
            batch_profiles['length'] = np.array(length, dtype=np.int32)
            for key in profiles[0].keys():
                if key == 'length':
                    continue
                value = [x[key] for x in profiles]
                padded_value = np.asarray(
                    [np.pad(a, (0, max_length - len(a)), 'constant', constant_values=0) for a in value], dtype=np.int32)
                batch_profiles[key] = padded_value
            return batch_profiles

        batch_size = self.batch_size
        total_num = len(rates)
        end = 0
        epoch = 0
        while epoch < epoch_num or epoch_num is None:
            start = end
            if start == total_num:
                epoch += 1
                start = 0
                if shuffle_every_epoch:
                    data = list(zip(users, items, rates))
                    shuffle(data)
                    users, items, rates = zip(*data)
            end = min(start + batch_size, total_num)
            batch_users = users[start:end]
            batch_items = items[start:end]
            batch_rates = np.array(rates[start:end], dtype=np.int32)
            batch_rates = batch_rates - 1  # shift from 1-5 to 0-4
            batch_user_profiles = parse_batch_profile(batch_users, self.user_profiles)
            batch_movie_profiles = parse_batch_profile(batch_items, self.movie_profiles)
            last_batch = end == total_num
            yield batch_user_profiles, batch_movie_profiles, batch_rates, last_batch

    def create_or_load_model(self, overwrite=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        variables = tf.global_variables()
        saver = tf.train.Saver(variables, max_to_keep=1)
        ckpt = tf.train.latest_checkpoint(self.ckpt_path)
        if ckpt is None or overwrite:
            print_with_time('fresh training... checkpoint path: %s' % self.ckpt_path)
            self.model_manager.remove_dir(self.ckpt_path)
            os.makedirs(self.ckpt_path, exist_ok=True)
            sess.run(tf.global_variables_initializer())
        else:
            print('load pre-training checkpoint at %s' % ckpt)
            saver.restore(sess, ckpt)
        self.saver = saver
        self.sess = sess

    # def filter_trainset(self, sample):
    #     user = sample[0]
    #     item = sample[1]
    #     self.user_profiles[user]

    def fit(self, trainset, valid_rate=0.1):
        valid_num = round(len(trainset) * valid_rate)
        train_num = len(trainset) - valid_num
        print_with_time("start training on %d samples, valid on %d samples" % (train_num, valid_num))
        users_train, items_train, rates_train = zip(*trainset[:train_num])
        users_valid, items_valid, rates_valid = zip(*trainset[train_num:])
        valid_input = list(zip(users_valid, items_valid))
        # Training loop
        records = []
        best_valid_loss = None
        last_save_epoch = None
        epoch = 0
        predictions = []
        labels = []
        for batch_user_profiles, batch_item_profiles, batch_rates, last_batch \
                in self.batch_iter(users_train, items_train, rates_train, True, self.epochs):
            feed_dict = self.make_feed_dict(batch_user_profiles, batch_item_profiles, batch_rates, 'train')
            _, step, loss, pred = self.sess.run(
                [self.train_op, self.global_step, self.model.loss, self.model.predictions], feed_dict=feed_dict)
            predictions.extend(pred)
            labels.extend(batch_rates)
            if step % 100 == 0:
                print_with_time("step {0} : loss = {1}".format(step, loss))
            if last_batch:
                epoch += 1
                train_acc = accuracy_score(labels, predictions)
                print("train prediction rate: %s" % str(Counter(np.array(predictions) + 1)))
                labels.clear()
                predictions.clear()
                # evaluate
                valid_prediction, _, valid_loss = self.prediction(valid_input, rates_valid, False)
                valid_acc = accuracy_score(rates_valid, valid_prediction)
                print("valid prediction rate: %s" % str(Counter(valid_prediction)))
                print_with_time('epoch %d: train_loss %f, valid_loss %f, train_acc %f, valid_acc %f' % (
                    epoch, loss, valid_loss, train_acc, valid_acc))
                if best_valid_loss is None or valid_loss < best_valid_loss:
                    print_with_time('get a better one!')
                    best_valid_loss = valid_loss
                    self.saver.save(self.sess, os.path.join(self.ckpt_path, 'model.ckpt'), global_step=step)
                    last_save_epoch = epoch

                records.append({'step': int(step), 'train_loss': float(loss), 'train_acc': float(train_acc),
                                'valid_loss': float(valid_loss), 'valid_acc': float(valid_acc)})
                # early stop
                if last_save_epoch is not None and self.early_stop is not None and epoch - last_save_epoch > self.early_stop:
                    print_with_time('No loss decrease on valid set for %d epochs, stop training' % self.early_stop)
                    break
        # result
        self.model_manager.save_json(records, 'train_loss')
        print_with_time('training done')

    def make_feed_dict(self, batch_user_profiles, batch_item_profiles, batch_rates=None, mode='test'):
        feed_dict = {
            self.model.targets_user: batch_user_profiles['target'],
            self.model.descriptions_user: batch_user_profiles['description'],
            self.model.sentiments_user: batch_user_profiles['sentiment'],
            self.model.pair_freq_user: batch_user_profiles['freq'],
            self.model.targets_item: batch_item_profiles['target'],
            self.model.descriptions_item: batch_item_profiles['description'],
            self.model.sentiments_item: batch_item_profiles['sentiment'],
            self.model.pair_freq_item: batch_item_profiles['freq'],
            self.model.keep_prob: 1.0,
            self.model.training: False
        }
        if batch_rates is not None:
            feed_dict.update({self.model.rating: batch_rates})
        if mode == 'train':
            feed_dict.update({self.model.keep_prob: 1 - self.dropout, self.model.training: True})
        return feed_dict

    def prediction(self, test_input, test_labels=None, only_prediction=True):
        labels = test_labels
        if labels is None:
            labels = [0] * len(test_input)
        users, items = zip(*test_input)
        test_batches = self.batch_iter(users, items, labels, False)
        _prediction = []
        _loss = 0
        _logits = []
        for batch_user_profiles, batch_items_profiles, batch_rates, last_batch in test_batches:
            if test_labels is None:
                feed_dict = self.make_feed_dict(batch_user_profiles, batch_items_profiles)
                pred, logits = self.sess.run((self.model.predictions, self.model.logits), feed_dict=feed_dict)
            else:
                feed_dict = self.make_feed_dict(batch_user_profiles, batch_items_profiles, batch_rates)
                pred, loss, logits = self.sess.run((self.model.predictions, self.model.loss, self.model.logits),
                                                   feed_dict=feed_dict)
                _loss += loss * len(batch_rates)
            _prediction.extend(pred)
            _logits.extend(logits)
            if last_batch:
                break
        _loss /= len(test_input)
        _prediction = np.array(_prediction) + 1  # shift from 0-4 to 1-5
        if only_prediction:
            return _prediction
        else:
            return _prediction, _logits, _loss

    def clean(self):
        self.sess.close()
