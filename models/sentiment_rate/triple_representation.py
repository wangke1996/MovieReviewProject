import os
import numpy as np
import tensorflow as tf
from models.lib import CONFIG
from models.lib.data_helper import load_np_array, make_vocab_lookup, load_json_file, shuffle_data_list
from models.lib.io_helper import print_with_time


class TripleSentimentRating:
    def initialize(self, model_manager):
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
            self.user_profiles = model_manager.load_json('user_profiles')
        except OSError:
            self.user_profiles = list(map(lambda x: parse_profile(x, 'user'), self.user_list))
            model_manager.save_json(self.user_profiles, 'user_profiles')
        print_with_time('initial item profiles')
        try:
            self.item_profiles = model_manager.load_json('item_profiles')
        except OSError:
            self.item_profiles = list(map(lambda x: parse_profile(x, 'subject'), self.item_list))
            model_manager.save_json(self.item_profiles, 'item_profiles')
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

    def make_feed_dict(self, batch_users, batch_items, batch_rates=None, mode='test', dropout=None,
                       batch_user_profiles=None, batch_item_profiles=None):
        def parse_batch_profile(ids, profile_list, profiles=None):
            if profiles is None:
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

        batch_user_profiles = parse_batch_profile(batch_users, self.user_profiles, batch_user_profiles)
        batch_item_profiles = parse_batch_profile(batch_items, self.item_profiles, batch_item_profiles)
        feed_dict = {
            self.targets_user: batch_user_profiles['target'],
            self.descriptions_user: batch_user_profiles['description'],
            self.sentiments_user: batch_user_profiles['sentiment'],
            self.pair_freq_user: batch_user_profiles['freq'],
            self.targets_item: batch_item_profiles['target'],
            self.descriptions_item: batch_item_profiles['description'],
            self.sentiments_item: batch_item_profiles['sentiment'],
            self.pair_freq_item: batch_item_profiles['freq'],
            self.keep_prob: 1.0,
            self.training: False
        }
        if batch_rates is not None:
            feed_dict.update({self.rating: batch_rates})
        if mode == 'train' and dropout is not None:
            feed_dict.update({self.keep_prob: 1 - dropout, self.training: True})
        return feed_dict

    def __init__(self, user_list, item_list, config=None,
                 # word_embedding_size=128,
                 # model_layers=(128,),
                 # model_layers=(256, 128, 64),
                 ):
        self.user_list = user_list
        self.item_list = item_list
        if config is None:
            config = {}
        sentiment_category = config.get('sentiment_category', 3)
        sentiment_embedding_size = config.get('sentiment_embedding_size', 3)
        word_embedding_size = config.get('word_embedding_size', 16)
        model_layers = config.get('model_layers', (32, 16))
        l2 = config.get('l2', 0.1)
        embedding_initializer = config.get('embedding_initializer', 'glorot_uniform')
        num_class = config.get('num_class', 5)
        use_focal_loss = config.get('use_focal_loss', False)
        self.max_length = config.get('max_length', 50)
        self.user_profiles = None
        self.item_profiles = None
        self.target_word2id = make_vocab_lookup(CONFIG.target_word_list, unk_token='UNK')
        self.description_word2id = make_vocab_lookup(CONFIG.description_word_list, unk_token='UNK')
        self.sentiment_word2id = make_vocab_lookup(CONFIG.sentiment_category_list)
        self.target_num = len(self.target_word2id)
        self.description_num = len(self.description_word2id)
        self.embedding_size = word_embedding_size
        self.sentiment_embedding_size = sentiment_embedding_size

        self.targets_user = tf.placeholder(tf.int32, [None, None])  # batch * len_user
        self.descriptions_user = tf.placeholder(tf.int32, [None, None])  # batch * len_user
        self.sentiments_user = tf.placeholder(tf.int32, [None, None])  # batch * len_user
        self.pair_freq_user = tf.placeholder(tf.int32, [None, None])  # batch * len_user
        # self.len_user = tf.placeholder(tf.int32, [None])  # 1 * batch

        self.targets_item = tf.placeholder(tf.int32, [None, None])  # batch * len_item
        self.descriptions_item = tf.placeholder(tf.int32, [None, None])  # batch * len_item
        self.sentiments_item = tf.placeholder(tf.int32, [None, None])  # batch * len_item
        self.pair_freq_item = tf.placeholder(tf.int32, [None, None])  # batch * len_item
        # self.len_item = tf.placeholder(tf.int32, [None])  # 1 * batch

        self.rating = tf.placeholder(tf.int32, None)  # 1 * batch

        self.training = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.targets_user)[0]

        with tf.variable_scope("embedding"):
            embeddings_target = tf.keras.layers.Embedding(
                self.target_num,
                word_embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                name="embedding_target")
            embeddings_description = tf.keras.layers.Embedding(
                self.description_num,
                word_embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                name="embedding_description")
            embeddings_sentiment = tf.keras.layers.Embedding(
                sentiment_category,
                sentiment_embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                name="embedding_sentiment")
            target_emb_user = embeddings_target(self.targets_user)
            description_emb_user = embeddings_description(self.descriptions_user)
            sentiment_emb_user = embeddings_sentiment(self.sentiments_user)
            target_emb_item = embeddings_target(self.targets_item)
            description_emb_item = embeddings_description(self.descriptions_item)
            sentiment_emb_item = embeddings_sentiment(self.sentiments_item)
            # init_embeddings_target = tf.random_uniform([target_word_num, word_embedding_size])
            # init_embeddings_description = tf.random_uniform([description_word_num, word_embedding_size])
            # init_embeddings_sentiment = tf.random_uniform([sentiment_category, sentiment_embedding_size])
            # embeddings_target = tf.get_variable("embeddings_target", initializer=init_embeddings_target)
            # embeddings_description = tf.get_variable("embeddings_description", initializer=init_embeddings_description)
            # embeddings_sentiment = tf.get_variable("embeddings_sentiment", initializer=init_embeddings_sentiment)
            # # user emb
            # target_emb_user = tf.nn.embedding_lookup(embeddings_target, self.targets_user)
            # description_emb_user = tf.nn.embedding_lookup(embeddings_description, self.descriptions_user)
            # sentiment_emb_user = tf.nn.embedding_lookup(embeddings_sentiment, self.sentiments_user)
            # # item emb
            # target_emb_item = tf.nn.embedding_lookup(embeddings_target, self.targets_item)
            # description_emb_item = tf.nn.embedding_lookup(embeddings_description, self.descriptions_item)
            # sentiment_emb_item = tf.nn.embedding_lookup(embeddings_sentiment, self.sentiments_item)

        with tf.variable_scope("encoding"):
            emb_user = tf.concat((target_emb_user, description_emb_user, sentiment_emb_user), axis=-1)
            emb_item = tf.concat((target_emb_item, description_emb_item, sentiment_emb_item), axis=-1)
            # weighted average by freq, todo: try attention instead
            encode_user = tf.transpose(
                tf.multiply(tf.transpose(tf.cast(self.pair_freq_user, tf.float32)), tf.transpose(emb_user)))
            encode_user = tf.reduce_sum(encode_user, axis=1)
            pair_freq_user_sum = tf.reduce_sum(self.pair_freq_user, axis=-1)
            pair_freq_user_sum = tf.where(tf.equal(pair_freq_user_sum, 0), tf.ones([self.batch_size], dtype=tf.int32),
                                          pair_freq_user_sum)
            encode_user = tf.transpose(tf.multiply(tf.transpose(encode_user), 1 / tf.cast(pair_freq_user_sum,
                                                                                          dtype=tf.float32)))  # batch * (2 * word_embedding_size + sentiment_embeddign_size)

            encode_item = tf.transpose(
                tf.multiply(tf.transpose(tf.cast(self.pair_freq_item, tf.float32)), tf.transpose(emb_item)))
            encode_item = tf.reduce_sum(encode_item, axis=1)

            pair_freq_item_sum = tf.reduce_sum(self.pair_freq_item, axis=-1)
            pair_freq_item_sum = tf.where(tf.equal(pair_freq_item_sum, 0), tf.ones([self.batch_size], dtype=tf.int32),
                                          pair_freq_item_sum)
            encode_item = tf.transpose(
                tf.multiply(tf.transpose(encode_item), 1 / tf.cast(pair_freq_item_sum,
                                                                   dtype=tf.float32)))  # batch * (2 * word_embedding_size + sentiment_embeddign_size)

        encode_output = tf.concat((encode_user, encode_item),
                                  axis=-1)  # batch * (4 * word_embedding_size + 2 * sentiment_embeddign_size)
        fc_output = encode_output
        num_layer = len(model_layers)  # Number of layers in the MLP
        for layer in range(1, num_layer):
            model_layer = tf.keras.layers.Dense(
                model_layers[layer],
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                activation='relu',
                name='FCL')
            fc_output = model_layer(fc_output)
            # batch_norm = tf.keras.layers.BatchNormalization()
            # fc_output = batch_norm(fc_output, training=self.training)
            # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, batch_norm.updates)
            # activation = tf.keras.activations.relu
            # fc_output = activation(fc_output)
            fc_output = tf.nn.dropout(fc_output, self.keep_prob)
        self.final_feature = fc_output
        with tf.name_scope("output"):
            self.logits = tf.keras.layers.Dense(num_class, activation=None, kernel_initializer="lecun_uniform",
                                                name="logits")(self.final_feature)
            # self.logits = tf.layers.dense(fc_output, num_class, activation=tf.nn.relu)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            if use_focal_loss:
                self.loss = tf.reduce_mean(focal_loss_softmax(logits=self.logits, labels=self.rating))
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.rating))


def focal_loss_softmax(labels, logits, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
    labels = tf.one_hot(labels, depth=y_pred.shape[1])
    loss = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    loss = tf.reduce_sum(loss, axis=1)
    return loss
