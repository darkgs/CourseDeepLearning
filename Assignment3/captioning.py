# This library is used for Assignment3_Part2_ImageCaptioning

# Write your own image captiong code
# You can modify the class structure
# and add additional function needed for image captionging

import tensorflow as tf
import numpy as np


class Captioning():
    def __init__(self, img_feature_dim, max_seq_len, n_words, rnn_hidden_dim, num_of_layers):

        self._config = {
            'img_feature_dim': img_feature_dim,
            'max_seq_len': max_seq_len,
            'n_words': n_words,
            'rnn_hidden_dim': rnn_hidden_dim,
            'num_of_layers': num_of_layers,
            'word_embed_dim': 512,
        }

        self._cell = None

    def build_model(self):
        max_seq_len = self._config['max_seq_len']
        img_feature_dim = self._config['img_feature_dim']
        n_words = self._config['n_words']
        rnn_hidden_dim = self._config['rnn_hidden_dim']
        num_of_layers = self._config['num_of_layers']
        word_embed_dim = self._config['word_embed_dim']

        word_embed_dim = n_words

        input_sequences = tf.placeholder(tf.int32, [None, max_seq_len])
        input_sequence_lens = tf.placeholder(tf.int32, [None])
        self._input_img_features = tf.placeholder(tf.float32, [None, img_feature_dim])
        drop_out_keep_rate = tf.placeholder_with_default(1.0, shape=())

        # idx2embedding
#self._embedding = tf.Variable(tf.random_normal([n_words, word_embed_dim]), dtype=tf.float32)
#rnn_input = tf.nn.embedding_lookup(self._embedding, input_sequences)
        rnn_input = tf.one_hot(input_sequences, n_words)

        # RNN cell
        def _single_cell():
            _cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_dim)
            return tf.contrib.rnn.DropoutWrapper(_cell, output_keep_prob=drop_out_keep_rate)

        self._cell = tf.nn.rnn_cell.MultiRNNCell([_single_cell() for _ in range(num_of_layers)])
        img2hidden = tf.layers.dense(self._input_img_features, rnn_hidden_dim)
        self._rnn_init = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                tf.matmul(self._input_img_features, tf.zeros((self._input_img_features.shape[1], rnn_hidden_dim))),
                img2hidden) for _ in range(num_of_layers)])

        rnn_outputs, state = tf.nn.dynamic_rnn(cell=self._cell,
                inputs=rnn_input,
                initial_state=self._rnn_init,
                sequence_length=input_sequence_lens,
                dtype=tf.float32)

        self._w_hidden2word = tf.get_variable('rnn_out_w', shape=(rnn_hidden_dim, word_embed_dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self._b_hidden2word = tf.get_variable('rnn_out_b', shape=(word_embed_dim), dtype=tf.float32)

#logits = tf.layers.dense(rnn_outputs, word_embed_dim)
        logits = tf.matmul(tf.reshape(rnn_outputs, [-1,rnn_hidden_dim]), self._w_hidden2word) \
                 + self._b_hidden2word
        logits = tf.reshape(logits, [-1, max_seq_len, word_embed_dim])

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=rnn_input[:,1:,:], logits=logits[:,:-1,:])
        loss = tf.reduce_mean(cross_entropy)

        optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        teacher_force = tf.argmax(logits, axis=2)

        return input_sequences, input_sequence_lens, self._input_img_features, loss, optimize, drop_out_keep_rate, teacher_force

    def predict(self):
        if self._cell == None:
            self.build_model()

        max_seq_len = self._config['max_seq_len']
        n_words = self._config['n_words']
        rnn_hidden_dim = self._config['rnn_hidden_dim']

        inputs = tf.matmul(tf.cast(self._input_img_features, dtype=tf.int32), 
                tf.zeros((self._input_img_features.shape[1], 1), dtype=tf.int32)) + 1
        inputs = tf.squeeze(tf.one_hot(inputs, n_words), axis=1)

        state = self._rnn_init
        predicts = []
        for i in range(max_seq_len):
            outputs, state = self._cell(inputs, state)
            logits = tf.matmul(tf.reshape(outputs, [-1,rnn_hidden_dim]), self._w_hidden2word) \
                     + self._b_hidden2word
            predict = tf.argmax(logits, axis=1)
#inputs = tf.nn.embedding_lookup(self._embedding, predict)
            inputs = tf.one_hot(predict, n_words)
            predicts.append(predict)

        return tf.transpose(tf.stack(predicts), [1,0])


