import numpy as np
import random
import tensorflow as tf
from models.lib import CONFIG
from models.lib.data_helper import make_vocab_lookup, load_json_file, make_profiles
from models.lib.io_helper import print_with_time


class TripleRepresentationV2:
    def triples_to_ids(self, triples):
        return list(map(lambda triple: (
            self.target_word2id.get(triple[0], 0),
            self.description_word2id.get(triple[1], 0),
            self.sentiment_word2id.get(triple[2], 0)
        ), triples))

    def initialize(self, model_manager):
        print_with_time('initialize profiles')
        user_profiles, item_profiles = make_profiles(False)

        print_with_time('initial user profiles')
        try:
            self.all_user_profiles = model_manager.load_json('user_profiles_v2')
        except OSError:
            self.all_user_profiles = list(map(self.triples_to_ids, user_profiles))
            model_manager.save_json(self.all_user_profiles, 'user_profiles_v2')
        print_with_time('initial item profiles')
        try:
            self.all_item_profiles = model_manager.load_json('item_profiles_v2')
        except OSError:
            self.all_item_profiles = list(map(self.triples_to_ids, item_profiles))
            model_manager.save_json(self.all_item_profiles, 'item_profiles_v2')
        print_with_time('profiles initialized')

    def sample_profile(self, _id, profile_list, triples=None):
        if triples is None:
            # get triples by id
            triples = profile_list[_id]
        if self.max_length is not None and len(triples) > self.max_length:
            triples = random.sample(triples, self.max_length)
        triples = self.triples_to_ids(triples)
        if len(triples) == 0:
            return (), (), ()
        return tuple(zip(*triples))

    def parse_batch_profiles(self, ids, profile_list, profiles=None):
        if ids is None:
            ids = [None] * len(profiles)
        if profiles is None:
            profiles = [None] * len(ids)
        profiles = map(lambda _id, triples: self.sample_profile(_id, profile_list, triples), ids, profiles)
        targets, descriptions, sentiments = list(zip(*profiles))
        padding = lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, padding='post')
        targets = padding(targets)
        descriptions = padding(descriptions)
        sentiments = padding(sentiments)
        return targets, descriptions, sentiments

    def make_feed_dict(self, batch_users, batch_items, batch_rates=None, mode='test', dropout=None,
                       batch_user_profiles=None, batch_item_profiles=None):
        batch_user_targets, batch_user_descriptions, batch_user_sentiments = \
            self.parse_batch_profiles(batch_users, self.all_user_profiles, batch_user_profiles)
        batch_item_targets, batch_item_descriptions, batch_item_sentiments = \
            self.parse_batch_profiles(batch_items, self.all_item_profiles, batch_item_profiles)

        feed_dict = {
            self.user_targets: batch_user_targets,
            self.item_targets: batch_item_targets,
            self.user_descriptions: batch_user_descriptions,
            self.item_descriptions: batch_item_descriptions,
            self.user_sentiments: batch_user_sentiments,
            self.item_sentiments: batch_item_sentiments,
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
        mf_dim_target = config.get('mf_dim_target', 8)
        mf_dim_description = config.get('mf_dim_description', 8)
        mf_dim_sentiment = config.get('mf_dim_sentiment', 2)
        target_embedding_size = config.get('target_embedding_size', 16)
        description_embedding_size = config.get('description_embedding_size', 16)
        sentiment_embedding_size = config.get('sentiment_embedding_size', 4)
        model_layers = config.get('model_layers', (32, 16))
        l2 = config.get('l2', 0.1)
        embedding_initializer = config.get('embedding_initializer', 'glorot_uniform')
        num_class = config.get('num_class', 5)
        self.max_length = config.get('max_length', 50)
        self.all_user_profiles = None
        self.all_item_profiles = None
        self.target_word2id = make_vocab_lookup(CONFIG.target_word_list, unk_token='UNK')
        self.description_word2id = make_vocab_lookup(CONFIG.description_word_list, unk_token='UNK')
        self.sentiment_word2id = make_vocab_lookup(CONFIG.sentiment_category_list)
        target_num = len(self.target_word2id)
        description_num = len(self.description_word2id)
        sentiment_num = len(self.sentiment_word2id)

        self.user_targets = tf.placeholder(tf.int32, [None, None])  # batch * len_user
        self.user_descriptions = tf.placeholder(tf.int32, [None, None])
        self.user_sentiments = tf.placeholder(tf.int32, [None, None])
        self.item_targets = tf.placeholder(tf.int32, [None, None])  # batch * len_item
        self.item_descriptions = tf.placeholder(tf.int32, [None, None])
        self.item_sentiments = tf.placeholder(tf.int32, [None, None])
        self.rating = tf.placeholder(tf.int32, None)
        self.training = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.user_targets)[0]

        with tf.variable_scope("embedding"):
            embeddings_target = tf.keras.layers.Embedding(
                target_num,
                target_embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                mask_zero=True,
                name="embedding_target")
            embeddings_description = tf.keras.layers.Embedding(
                description_num,
                description_embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                mask_zero=True,
                name="embedding_description")
            embeddings_sentiment = tf.keras.layers.Embedding(
                sentiment_num,
                sentiment_embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                mask_zero=True,
                name="embedding_sentiment")
            embedding_user_target = embeddings_target(self.user_targets)
            embedding_item_target = embeddings_target(self.item_targets)
            embedding_user_description = embeddings_description(self.user_descriptions)
            embedding_item_description = embeddings_description(self.item_descriptions)
            embedding_user_sentiment = embeddings_sentiment(self.user_sentiments)
            embedding_item_sentiment = embeddings_sentiment(self.item_sentiments)

            def average_with_weights(emb, weights):
                emb = tf.transpose(tf.multiply(tf.transpose(tf.cast(weights, tf.float32)), tf.transpose(emb)))
                emb = tf.reduce_sum(emb, axis=1)
                weights_sum = tf.reduce_sum(weights, axis=-1)
                weights_sum = tf.where(tf.equal(weights_sum, 0), tf.ones([self.batch_size], dtype=tf.int32),
                                       weights_sum)
                emb = tf.transpose(tf.multiply(tf.transpose(emb), 1 / tf.cast(weights_sum, dtype=tf.float32)))
                return emb

            embedding_user_target, embedding_item_target, embedding_user_description, embedding_item_description, \
            embedding_user_sentiment, embedding_item_sentiment = map(
                lambda x: average_with_weights(x, tf.cast(x._keras_mask, dtype=tf.int32)), (
                    embedding_user_target, embedding_item_target, embedding_user_description,
                    embedding_item_description, embedding_user_sentiment, embedding_item_sentiment))

        with tf.variable_scope("latent"):
            def split_mf(emb, mf_dim, _type):
                mf_part = tf.keras.layers.Lambda(lambda x: x[:, :mf_dim], name="embedding_mf_%s" % _type)(emb)
                mlp_part = tf.keras.layers.Lambda(lambda x: x[:, mf_dim:], name="embedding_mlp_%s" % _type)(emb)
                return mf_part, mlp_part

            # GMF part
            mf_user_target, mlp_user_target = split_mf(embedding_user_target, mf_dim_target, 'user_target')
            mf_user_description, mlp_user_description = split_mf(embedding_user_description, mf_dim_description,
                                                                 'user_description')
            mf_user_sentiment, mlp_user_sentiment = split_mf(embedding_user_sentiment, mf_dim_sentiment,
                                                             'user_sentiment')
            mf_item_target, mlp_item_target = split_mf(embedding_item_target, mf_dim_target, 'item_target')
            mf_item_description, mlp_item_description = split_mf(embedding_item_description, mf_dim_description,
                                                                 'item_description')
            mf_item_sentiment, mlp_item_sentiment = split_mf(embedding_item_sentiment, mf_dim_sentiment,
                                                             'item_sentiment')

            mf_user_latent = tf.concat([mf_user_target, mf_user_description, mf_user_sentiment], axis=-1)
            mf_item_latent = tf.concat([mf_item_target, mf_item_description, mf_item_sentiment], axis=-1)

            mlp_user_latent = tf.concat([mlp_user_target, mlp_user_description, mlp_user_sentiment], axis=-1)
            mlp_item_latent = tf.concat([mlp_item_target, mlp_item_description, mlp_item_sentiment], axis=-1)

            # Element-wise multiply
            mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

            # Concatenation of two latent features
            mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

            num_layer = len(model_layers)  # Number of layers in the MLP
            for layer in range(0, num_layer):
                model_layer = tf.keras.layers.Dense(
                    model_layers[layer],
                    kernel_regularizer=tf.keras.regularizers.l2(l2),
                    activation="relu",
                    name='FCL')
                mlp_vector = model_layer(mlp_vector)
                mlp_vector = tf.nn.dropout(mlp_vector, self.keep_prob)

        # Concatenate GMF and MLP parts
        self.final_feature = tf.keras.layers.concatenate([mf_vector, mlp_vector])

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(self.final_feature, num_class, activation=None)
            # self.logits = tf.layers.dense(fc_output, num_class, activation=tf.nn.relu)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.rating))
