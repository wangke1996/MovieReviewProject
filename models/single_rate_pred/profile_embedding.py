import random
import tensorflow as tf
from models.lib import CONFIG
from models.lib.data_helper import make_vocab_lookup


class ProfileEmbedding:
    def make_feed_dict(self, batch_reviews, batch_profiles, mode='test', dropout=None):
        def sample_profile(triples):
            if self.max_length is not None and len(triples) > self.max_length:
                triples = random.sample(triples, self.max_length)
            triples = [
                (self.target_word2id.get(t, 0), self.description_word2id.get(d, 0), self.sentiment_word2id.get(s, 0))
                for t, d, s in triples]
            if len(triples) == 0:
                return (), (), ()
            return tuple(zip(*triples))

        profiles = tuple(map(lambda x: sample_profile(x), batch_profiles))
        targets, descriptions, sentiments = list(zip(*profiles))
        padding = lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, padding='post')
        targets = padding(targets)
        descriptions = padding(descriptions)
        sentiments = padding(sentiments)
        feed_dict = {
            self.targets: targets,
            self.descriptions: descriptions,
            self.sentiments: sentiments,
            self.keep_prob: 1.0,
            self.training: False
        }
        if mode == 'train' and dropout is not None:
            feed_dict.update({self.keep_prob: 1 - dropout, self.training: True})
        return feed_dict

    def __init__(self, config=None,
                 # word_embedding_size=128,
                 # model_layers=(128,),
                 # model_layers=(256, 128, 64),
                 ):
        if config is None:
            config = {}
        target_embedding_size = config.get('target_embedding_size', 16)
        description_embedding_size = config.get('description_embedding_size', 16)
        sentiment_embedding_size = config.get('sentiment_embedding_size', 4)
        l2 = config.get('l2', 0.1)
        embedding_initializer = config.get('embedding_initializer', 'glorot_uniform')
        self.max_length = config.get('max_length', 10)
        self.target_word2id = make_vocab_lookup(CONFIG.target_word_list, unk_token='UNK')
        self.description_word2id = make_vocab_lookup(CONFIG.description_word_list, unk_token='UNK')
        self.sentiment_word2id = make_vocab_lookup(CONFIG.sentiment_category_list)
        target_num = len(self.target_word2id)
        description_num = len(self.description_word2id)
        sentiment_num = len(self.sentiment_word2id)

        self.targets = tf.placeholder(tf.int32, [None, None])  # batch * len_user
        self.descriptions = tf.placeholder(tf.int32, [None, None])
        self.sentiments = tf.placeholder(tf.int32, [None, None])
        self.training = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.targets)[0]

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
            target_emb = embeddings_target(self.targets)
            description_emb = embeddings_description(self.descriptions)
            sentiment_emb = embeddings_sentiment(self.sentiments)

            def average_with_weights(emb, weights):
                emb = tf.transpose(tf.multiply(tf.transpose(tf.cast(weights, tf.float32)), tf.transpose(emb)))
                emb = tf.reduce_sum(emb, axis=1)
                weights_sum = tf.reduce_sum(weights, axis=-1)
                weights_sum = tf.where(tf.equal(weights_sum, 0), tf.ones([self.batch_size], dtype=tf.int32),
                                       weights_sum)
                emb = tf.transpose(tf.multiply(tf.transpose(emb), 1 / tf.cast(weights_sum, dtype=tf.float32)))
                return emb

            target_emb, description_emb, sentiment_emb, = map(
                lambda x: average_with_weights(x, tf.cast(x._keras_mask, dtype=tf.int32)),
                (target_emb, description_emb, sentiment_emb))

        self.final_feature = tf.keras.layers.concatenate([target_emb, description_emb, sentiment_emb])
