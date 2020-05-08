import tensorflow as tf
import os
import numpy as np
import math
from models.lib import ModelManager, CONFIG
from models.lib.io_helper import load_np_array, print_with_time
from random import shuffle
from sklearn.metrics import accuracy_score


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


class NeuralCF:
    def __init__(self, user_list, item_list, config=None):
        if config is None:
            config = {}
        mf_dim = config.get('mf_dim', 16)
        model_layers = config.get('model_layers', (32, 16))
        mlp_reg_layers = config.get('mlp_reg_layers', (0.1, 0.1))
        embedding_initializer = config.get('embedding_initializer', 'glorot_uniform')
        mf_regularization = config.get('mf_regularization', 0.1)
        num_class = config.get('num_class', 5)
        self.user_list = user_list
        self.item_list = item_list
        self.users = tf.placeholder(tf.int32, [None])
        self.items = tf.placeholder(tf.int32, [None])
        self.rating = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        embedding_user = tf.keras.layers.Embedding(
            len(self.user_list),
            mf_dim + model_layers[0] // 2,
            embeddings_initializer=embedding_initializer,
            embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
            input_length=1,
            name="embedding_user")(
            self.users)

        embedding_item = tf.keras.layers.Embedding(
            len(self.item_list),
            mf_dim + model_layers[0] // 2,
            embeddings_initializer=embedding_initializer,
            embeddings_regularizer=tf.keras.regularizers.l2(mf_regularization),
            input_length=1,
            name="embedding_item")(
            self.items)

        def mf_slice_fn(x):
            # x = tf.squeeze(x, [1])
            return x[:, :mf_dim]

        def mlp_slice_fn(x):
            # x = tf.squeeze(x, [1])
            return x[:, mf_dim:]

        # GMF part
        mf_user_latent = tf.keras.layers.Lambda(
            mf_slice_fn, name="embedding_user_mf")(embedding_user)
        mf_item_latent = tf.keras.layers.Lambda(
            mf_slice_fn, name="embedding_item_mf")(embedding_item)

        # MLP part
        mlp_user_latent = tf.keras.layers.Lambda(
            mlp_slice_fn, name="embedding_user_mlp")(embedding_user)
        mlp_item_latent = tf.keras.layers.Lambda(
            mlp_slice_fn, name="embedding_item_mlp")(embedding_item)

        # Element-wise multiply
        mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

        # Concatenation of two latent features
        mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

        num_layer = len(model_layers)  # Number of layers in the MLP
        for layer in range(1, num_layer):
            model_layer = tf.keras.layers.Dense(
                model_layers[layer],
                kernel_regularizer=tf.keras.regularizers.l2(mlp_reg_layers[layer]),
                activation="relu",
                name='FCL')
            mlp_vector = model_layer(mlp_vector)
            mlp_vector = tf.nn.dropout(mlp_vector, self.keep_prob)

        # Concatenate GMF and MLP parts
        self.final_feature = tf.keras.layers.concatenate([mf_vector, mlp_vector])

        # Final prediction layer
        self.logits = tf.keras.layers.Dense(num_class, activation=None, kernel_initializer="lecun_uniform",
                                            name="logits")(self.final_feature)
        self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.rating))

    def make_feed_dict(self, batch_users, batch_items, batch_rates=None, mode='test', dropout=None):
        feed_dict = {
            self.users: batch_users,
            self.items: batch_items,
            self.keep_prob: 1.0
        }
        if batch_rates is not None:
            feed_dict.update({self.rating: batch_rates})
        if mode == 'train' and dropout is not None:
            feed_dict.update({self.keep_prob: 1 - dropout})
        return feed_dict


class NCF:
    def __init__(self, model_dir, batch_size=8192, epochs=50, lr=1e-3, early_stop=10, dropout=0.5, overwrite=True,
                 **kwargs):
        self.model_manager = ModelManager(model_dir)
        self.ckpt_path = os.path.join(self.model_manager.path_name, 'ckpt')
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stop = early_stop
        self.dropout = dropout
        self.user_list = load_np_array(CONFIG.user_list_file)
        self.item_list = load_np_array(CONFIG.movie_list_file)
        tf.reset_default_graph()
        self.model = NeuralCF(len(self.user_list), len(self.item_list), **kwargs)
        self.sess = None
        self.saver = None
        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope("Optimizer"):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.model.loss, params)
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
            _, step, loss = self.sess.run([self.train_op, self.global_step, self.model.loss], feed_dict=feed_dict)
            if step % 100 == 0:
                print_with_time("step {0} : loss = {1}".format(step, loss))
            if last_batch:
                epoch += 1
                # evaluate
                valid_prediction, _, valid_loss = self.prediction(valid_input, rates_valid, False)
                valid_acc = accuracy_score(rates_valid, valid_prediction)
                print_with_time('epoch %d: valid_loss %f, valid_acc %f' % (epoch, valid_loss, valid_acc))
                if best_valid_loss is None or valid_loss < best_valid_loss:
                    print_with_time('get a better one!')
                    best_valid_loss = valid_loss
                    self.saver.save(self.sess, os.path.join(self.ckpt_path, 'model.ckpt'), global_step=step)
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

    def make_feed_dict(self, batch_users, batch_items, batch_rates=None, mode='test'):
        feed_dict = {
            self.model.users: batch_users,
            self.model.items: batch_items,
            self.model.keep_prob: 1.0
        }
        if batch_rates is not None:
            feed_dict.update({self.model.rating: batch_rates})
        if mode == 'train':
            feed_dict.update({self.model.keep_prob: 1 - self.dropout})
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
        for batch_users, batch_items, batch_rates, last_batch in test_batches:
            if test_labels is None:
                feed_dict = self.make_feed_dict(batch_users, batch_items)
                pred, logits = self.sess.run((self.model.predictions, self.model.logits), feed_dict=feed_dict)
            else:
                feed_dict = self.make_feed_dict(batch_users, batch_items, batch_rates)
                pred, loss, logits = self.sess.run((self.model.predictions, self.model.loss, self.model.logits),
                                                   feed_dict=feed_dict)
                _loss += loss * self.batch_size
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
