import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from random import shuffle, sample
from models.lib import CONFIG, ModelManager
from models.lib.data_helper import load_np_array
from models.lib.io_helper import print_with_time
from .neural_CF import NeuralCF
from .triple_representation import TripleSentimentRating
from .triple_representation_v2 import TripleRepresentationV2


class UnionModel:
    def __init__(self, model_dir, batch_size=8192, epochs=250, lr=1e-2, early_stop=10, dropout=0.5, overwrite=True,
                 num_class=5, prediction_times=10, models=(NeuralCF, TripleSentimentRating), model_masks=(True, True),
                 pretrain_masks=(False, False), pretrain_dirs=(None, None), configs=(None, None), **kwargs):
        self.model_manager = ModelManager(model_dir)
        self.best_ckpt = None
        self.ckpt_path = os.path.join(self.model_manager.path_name, 'ckpt')
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.prediction_times = prediction_times
        self.early_stop = early_stop
        self.dropout = dropout
        self.user_list = load_np_array(CONFIG.user_list_file)
        self.item_list = load_np_array(CONFIG.movie_list_file)
        self.models = []
        self.pretrain_dirs = []
        tf.reset_default_graph()
        self.rating = tf.placeholder(tf.int32, [None])
        for model, mask, pretrain_mask, pretrain_dir, config in zip(models, model_masks, pretrain_masks, pretrain_dirs,
                                                                    configs):
            if mask is False:
                continue
            with tf.variable_scope(model.__name__):
                m = model(self.user_list, self.item_list, config)
                if 'initialize' in dir(model):
                    m.initialize(self.model_manager)
                self.models.append(m)
                self.pretrain_dirs.append(pretrain_dir if pretrain_mask else None)
        with tf.variable_scope("output"):
            self.final_feature = tf.concat([x.final_feature for x in self.models], axis=-1)
            self.logits = tf.keras.layers.Dense(num_class, activation=None, kernel_initializer="lecun_uniform",
                                                name="logits")(self.final_feature)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)
            self.soft_predictions = tf.squeeze(tf.matmul(tf.expand_dims(tf.range(num_class, dtype=tf.float32), 0),
                                                         tf.transpose(tf.nn.softmax(self.logits))))
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.rating))

        self.sess = None
        self.saver = None
        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope("Optimizer"):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        self.create_or_load_model(overwrite)

    def batch_iter(self, users, items, rates, shuffle_every_epoch=True, epoch_num=1):
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
            last_batch = end == total_num
            yield batch_users, batch_items, batch_rates, last_batch

    def make_feed_dict(self, batch_users, batch_items, batch_rates=None, mode='test', batch_user_profiles=None,
                       batch_item_profiles=None):
        feed_dict = {}
        for m in self.models:
            feed_dict.update(
                m.make_feed_dict(batch_users, batch_items, batch_rates, mode, self.dropout, batch_user_profiles,
                                 batch_item_profiles))
        if batch_rates is not None:
            feed_dict.update({self.rating: batch_rates})
        return feed_dict

    def create_or_load_model(self, overwrite=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.InteractiveSession(config=config)
        variables = tf.global_variables()
        saver = tf.train.Saver(variables, max_to_keep=1)
        ckpt = tf.train.latest_checkpoint(self.ckpt_path)
        if ckpt is None or overwrite:
            print_with_time('fresh training... checkpoint path: %s' % self.ckpt_path)
            self.model_manager.remove_dir(self.ckpt_path)
            os.makedirs(self.ckpt_path, exist_ok=True)
            sess.run(tf.global_variables_initializer())
            for m, pretrain_dir in zip(self.models, self.pretrain_dirs):
                if pretrain_dir is None:
                    continue
                pretrain_ckpt = tf.train.latest_checkpoint(pretrain_dir)
                if pretrain_ckpt is None:
                    print_with_time('no checkpoint found in %s' % pretrain_dir)
                else:
                    namespace = m.__class__.__name__ + '/'
                    sub_variables = [x for x in variables if x.name.startswith(namespace)]
                    sub_saver = tf.train.Saver(sub_variables)
                    sub_saver.restore(sess, pretrain_ckpt)
                    print_with_time('load %d variables from %s successfully' % (len(sub_variables), pretrain_dir))
        else:
            print('load pre-training checkpoint at %s' % ckpt)
            saver.restore(sess, ckpt)
        self.saver = saver
        self.sess = sess

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
        for batch_users, batch_items, batch_rates, last_batch \
                in self.batch_iter(users_train, items_train, rates_train, True, self.epochs):
            feed_dict = self.make_feed_dict(batch_users, batch_items, batch_rates, 'train')
            _, step, loss = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            if step % 100 == 0:
                print_with_time("step {0} : loss = {1}".format(step, loss))
            if last_batch:
                epoch += 1
                # evaluate
                valid_prediction, valid_loss = self.prediction(valid_input, rates_valid, False)
                valid_acc = accuracy_score(rates_valid, valid_prediction)
                print_with_time('epoch %d: valid_loss %f, valid_acc %f' % (epoch, valid_loss, valid_acc))
                if best_valid_loss is None or valid_loss < best_valid_loss:
                    print_with_time('get a better one!')
                    best_valid_loss = valid_loss
                    self.saver.save(self.sess, os.path.join(self.ckpt_path, 'model.ckpt'), global_step=step)
                    self.best_ckpt = os.path.join(self.ckpt_path, 'model.ckpt-%d' % step)
                    last_save_epoch = epoch
                records.append({'step': int(step), 'train_loss': float(loss), 'valid_loss': float(valid_loss),
                                'valid_acc': float(valid_acc)})
                # early stop
                if last_save_epoch is not None and self.early_stop is not None and epoch - last_save_epoch > self.early_stop:
                    print_with_time('No loss decrease on valid set for %d epochs, stop training' % self.early_stop)
                    break
        # result
        self.model_manager.save_json(records, 'train_loss')
        print_with_time('training done')

    def prediction(self, test_input, test_labels=None, only_prediction=True, from_bets_model=False):
        if from_bets_model:
            if self.best_ckpt is None:
                print_with_time('No checkpoint found! Use current weights!')
            else:
                self.saver.restore(self.sess, self.best_ckpt)
        predictions = []
        losses = 0
        for _ in range(self.prediction_times):
            res = self._prediction(test_input, test_labels, only_prediction)
            if only_prediction:
                predictions.append(res)
            else:
                predictions.append(res[0])
                losses += res[1]
        averaged_prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, minlength=2)), axis=0,
                                                  arr=predictions)
        averaged_loss = losses / self.prediction_times
        if only_prediction:
            return averaged_prediction
        else:
            return averaged_prediction, averaged_loss

    def _prediction(self, test_input, test_labels=None, only_prediction=True):

        labels = test_labels
        if labels is None:
            labels = [0] * len(test_input)
        users, items = zip(*test_input)
        test_batches = self.batch_iter(users, items, labels, False)
        _prediction = []
        _loss = 0
        for batch_users, batch_items, batch_rates, last_batch in test_batches:
            if test_labels is None:
                feed_dict = self.make_feed_dict(batch_users, batch_items)
                pred = self.sess.run(self.predictions, feed_dict=feed_dict)
            else:
                feed_dict = self.make_feed_dict(batch_users, batch_items, batch_rates)
                pred, loss = self.sess.run((self.predictions, self.loss), feed_dict=feed_dict)
                _loss += loss * len(pred)
            _prediction.extend(pred)
            if last_batch:
                break
        _loss /= len(test_input)
        _prediction = np.array(_prediction) + 1  # shift from 0-4 to 1-5
        if only_prediction:
            return _prediction
        else:
            return _prediction, _loss

    def recommend_movies(self, user=None, user_profile=None, candidate_movie_indexes=None, candidate_num=100,
                         recommend_num=10):
        for model in self.models:
            assert model.__class__ in [TripleRepresentationV2, TripleSentimentRating], \
                "unsupported model type for recommendation: %s" % model.__class__.__name__
        if candidate_movie_indexes is None:
            candidate_movie_indexes = sample(list(range(len(self.item_list))), candidate_num)
        candidate_num = len(candidate_movie_indexes)
        if user_profile is not None:
            user_profile = [user_profile for _ in range(candidate_num)]
        else:
            user = [user for _ in range(candidate_num)]
        feed_dict = self.make_feed_dict(user, candidate_movie_indexes, mode='test', batch_user_profiles=user_profile)
        scores = self.sess.run(self.soft_predictions, feed_dict=feed_dict)
        recommend_movie_index = np.argsort(-scores)[:recommend_num]
        result_index = np.array(candidate_movie_indexes)[recommend_movie_index]
        return list(self.item_list[result_index])

    def clean(self):
        self.sess.close()
