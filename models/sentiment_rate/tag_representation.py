import numpy as np
import random
import tensorflow as tf
from models.lib import CONFIG
from models.lib.data_helper import make_vocab_lookup, load_json_file, make_tags
from models.lib.io_helper import print_with_time


class TagRepresentation:
    def initialize(self, model_manager):
        print_with_time('initialize tags')
        user_tags, item_tags, _ = make_tags(False)

        def map_tags(tags):
            return list(map(lambda x: self.tag_word2id.get(x, 0), tags))

        self.all_user_tags = list(map(map_tags, user_tags))
        self.all_item_tags = list(map(map_tags, item_tags))
        print_with_time('tags initialized')

    def make_feed_dict(self, batch_users, batch_items, batch_rates=None, mode='test', dropout=None):
        def sample_tags(_id, tags_list):
            tags = tags_list[_id]
            if self.max_length is not None and len(tags) > self.max_length:
                tags = random.sample(tags, self.max_length)
            tags = list(map(lambda x: self.tag_word2id.get(x, 0), tags))
            return tags

        def parse_batch_tags(ids, tags_list):
            tags = list(map(lambda x: sample_tags(x, tags_list), ids))
            padded_tags = tf.keras.preprocessing.sequence.pad_sequences(tags, padding='post')
            return padded_tags

        batch_user_tags = parse_batch_tags(batch_users, self.all_user_tags)
        batch_item_tags = parse_batch_tags(batch_items, self.all_item_tags)

        feed_dict = {
            self.user_tags: batch_user_tags,
            self.item_tags: batch_item_tags,
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
        mf_dim = config.get('mf_dim', 8)
        tag_embedding_size = config.get('tag_embedding_size', 16)
        model_layers = config.get('model_layers', (32, 16))
        l2 = config.get('l2', 0.1)
        embedding_initializer = config.get('embedding_initializer', 'glorot_uniform')
        num_class = config.get('num_class', 5)
        self.max_length = config.get('max_length', 50)
        self.all_user_tags = None
        self.all_item_tags = None
        self.tag_word2id = make_vocab_lookup(CONFIG.tag_word_list, unk_token='UNK')
        tag_num = len(self.tag_word2id)

        self.user_tags = tf.placeholder(tf.int32, [None, None])  # batch * len_user
        self.item_tags = tf.placeholder(tf.int32, [None, None])  # batch * len_item
        self.rating = tf.placeholder(tf.int32, None)  # 1 * batch

        self.training = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.user_tags)[0]

        with tf.variable_scope("embedding"):
            embeddings_tag = tf.keras.layers.Embedding(
                tag_num,
                tag_embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                mask_zero=True,
                name="embedding_tag")
            embedding_user = embeddings_tag(self.user_tags)
            embedding_item = embeddings_tag(self.item_tags)

            def average_with_weights(emb, weights):
                emb = tf.transpose(tf.multiply(tf.transpose(tf.cast(weights, tf.float32)), tf.transpose(emb)))
                emb = tf.reduce_sum(emb, axis=1)
                weights_sum = tf.reduce_sum(weights, axis=-1)
                weights_sum = tf.where(tf.equal(weights_sum, 0), tf.ones([self.batch_size], dtype=tf.int32),
                                       weights_sum)
                emb = tf.transpose(tf.multiply(tf.transpose(emb), 1 / tf.cast(weights_sum, dtype=tf.float32)))
                return emb

            embedding_user = average_with_weights(embedding_user, tf.cast(embedding_user._keras_mask, dtype=tf.int32))
            embedding_item = average_with_weights(embedding_item, tf.cast(embedding_item._keras_mask, dtype=tf.int32))
        with tf.variable_scope("latent"):
            # GMF part
            mf_user_latent = tf.keras.layers.Lambda(lambda x: x[:, :mf_dim], name="embedding_user_mf")(embedding_user)
            mf_item_latent = tf.keras.layers.Lambda(lambda x: x[:, :mf_dim], name="embedding_item_mf")(embedding_item)

            # MLP part
            mlp_user_latent = tf.keras.layers.Lambda(lambda x: x[:, mf_dim:], name="embedding_user_mlp")(embedding_user)
            mlp_item_latent = tf.keras.layers.Lambda(lambda x: x[:, mf_dim:], name="embedding_item_mlp")(embedding_item)

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
