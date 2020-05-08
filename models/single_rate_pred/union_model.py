import os
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import accuracy_score
from random import shuffle
from models.lib.io_helper import print_with_time
from models.lib.utils import ModelManager
from .rnn import RNN


class UnionModel:
    def __init__(self, model_dir, batch_size=8192, epochs=250, lr=1e-3, early_stop=10, dropout=0.5, overwrite=True,
                 num_class=5, prediction_times=1, models=(RNN,), model_masks=(True),
                 pretrain_masks=(False,), pretrain_dirs=(None,), configs=(None), **kwargs):
        self.model_manager = ModelManager(model_dir)
        self.best_ckpt = None
        self.ckpt_path = os.path.join(self.model_manager.path_name, 'ckpt')
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.prediction_times = prediction_times
        self.early_stop = early_stop
        self.dropout = dropout
        self.models = []
        self.pretrain_dirs = []
        tf.reset_default_graph()
        self.rating = tf.placeholder(tf.int32, [None])
        for model, mask, pretrain_mask, pretrain_dir, config in zip(models, model_masks, pretrain_masks, pretrain_dirs,
                                                                    configs):
            if mask is False:
                continue
            with tf.variable_scope(model.__name__):
                m = model(config)
                if 'initialize' in dir(model):
                    m.initialize(self.model_manager)
                self.models.append(m)
                self.pretrain_dirs.append(pretrain_dir if pretrain_mask else None)
        with tf.variable_scope("output"):
            self.final_feature = tf.concat([x.final_feature for x in self.models], axis=-1)
            self.logits = tf.keras.layers.Dense(num_class, activation=None, kernel_initializer="lecun_uniform",
                                                name="logits")(self.final_feature)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)
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

    def batch_iter(self, reviews, profiles, rates, shuffle_every_epoch=True, epoch_num=1):
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
                    data = list(zip(reviews, profiles, rates))
                    shuffle(data)
                    reviews, profiles, rates = zip(*data)
            end = min(start + batch_size, total_num)
            batch_reviews = reviews[start:end]
            batch_profiles = profiles[start:end]
            batch_rates = np.array(rates[start:end], dtype=np.int32)
            batch_rates = batch_rates - 1  # shift from 1-5 to 0-4
            last_batch = end == total_num
            yield batch_reviews, batch_profiles, batch_rates, last_batch

    def make_feed_dict(self, batch_reviews, batch_profiles, batch_rates=None, mode='test'):
        feed_dict = {}
        for m in self.models:
            feed_dict.update(m.make_feed_dict(batch_reviews, batch_profiles, mode, self.dropout))
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
        reviews_train, profiles_train, rates_train = zip(*trainset[:train_num])
        reviews_valid, profiles_valid, rates_valid = zip(*trainset[train_num:])
        valid_input = list(zip(reviews_valid, profiles_valid))
        # Training loop
        records = []
        best_valid_loss = None
        last_save_epoch = None
        epoch = 0
        for batch_reviews, batch_profiles, batch_rates, last_batch \
                in self.batch_iter(reviews_train, profiles_train, rates_train, True, self.epochs):
            feed_dict = self.make_feed_dict(batch_reviews, batch_profiles, batch_rates, 'train')
            _, step, loss = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            if step % 100 == 0:
                print_with_time("step {0} : loss = {1}".format(step, loss))
            if last_batch:
                epoch += 1
                # evaluate
                valid_prediction, valid_loss = self.prediction(valid_input, rates_valid, False)
                valid_acc = accuracy_score(rates_valid, valid_prediction)
                print_with_time('epoch %d: valid_loss %f, valid_acc %f' % (epoch, valid_loss, valid_acc))
                print_with_time(str(Counter(valid_prediction)))
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
        reviews, profiles = zip(*test_input)
        test_batches = self.batch_iter(reviews, profiles, labels, False)
        _prediction = []
        _loss = 0
        for batch_reviews, batch_profiles, batch_rates, last_batch in test_batches:
            if test_labels is None:
                feed_dict = self.make_feed_dict(batch_reviews, batch_profiles)
                pred = self.sess.run(self.predictions, feed_dict=feed_dict)
            else:
                feed_dict = self.make_feed_dict(batch_reviews, batch_profiles, batch_rates)
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

    def clean(self):
        self.sess.close()
