import tensorflow as tf
from models.lib.data_helper import make_vocab_lookup
from models.lib import CONFIG


class RNN(object):
    def parse_review(self, review):
        words = [x for x in ' '.join(review).split(' ') if x != '']
        ids = [self.word2id.get(x, 0) for x in words]
        return ids

    def make_feed_dict(self, batch_reviews, batch_profiles=None, mode='test', dropout=None):
        reviews = list(map(self.parse_review, batch_reviews))
        lens = list(map(len, reviews))
        maxlen = min(self.max_length, max(lens))
        reviews = tf.keras.preprocessing.sequence.pad_sequences(reviews, maxlen=maxlen, padding='post',
                                                                truncating='post')
        feed_dict = {
            self.x: reviews,
            self.keep_prob: 1.0
        }
        if mode == 'train' and dropout is not None:
            feed_dict.update({self.keep_prob: 1 - dropout})
        return feed_dict

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.max_length = config.get('max_length', 50)
        embedding_size = config.get('embedding_size', 256)
        embedding_initializer = config.get('embedding_initializer', 'glorot_uniform')
        hidden_layer_num = config.get('hidden_layer_num', 1)
        num_hidden = config.get('num_hidden', 50)
        l2 = config.get('l2', 0.1)
        self.word2id = make_vocab_lookup(CONFIG.vocab_file, unk_token='UNK')
        word_num = len(self.word2id)
        self.x = tf.placeholder(tf.float32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.x)[0]
        with tf.variable_scope("embedding"):
            embeddings = tf.keras.layers.Embedding(
                word_num,
                embedding_size,
                embeddings_initializer=embedding_initializer,
                embeddings_regularizer=tf.keras.regularizers.l2(l2),
                input_length=1,
                mask_zero=True,
                name="embedding_tag")
            embedding_sequence = embeddings(self.x)
        with tf.variable_scope("rnn"):
            cells = [tf.keras.layers.LSTMCell(num_hidden) for _ in range(hidden_layer_num)]
            rnn_outputs = tf.keras.layers.RNN(cells)(embedding_sequence)
        self.final_feature = rnn_outputs
