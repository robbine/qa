"""Defines a base model to hold a common model interface.
"""

import tensorflow as tf

from abc import ABCMeta, abstractmethod
from model.input_util import *
from model.tf_util import convert_dense_to_sparse_tensor

class BaseModel(object):
    def __init__(self, options, sq_dataset, embeddings,
            word_chars, linear_interpolation, cove_cells, sess):
        self.sq_dataset = sq_dataset
        self.options = options
        self.num_words = self.sq_dataset.embeddings.shape[0]
        self.word_dim = self.sq_dataset.embeddings.shape[1]
        self.ctx_iterator, self.qst_iterator, self.ctx_pos_iterator, self.ctx_ner_iterator, \
        self.spn_iterator, self.data_index_iterator, self.qst_pos_iterator, self.qst_ner_iterator, \
        self.wiq_iterator, self.wic_iterator = sq_dataset.iterator.get_next()
        self.embeddings = embeddings
        self.word_chars = word_chars
        self.cove_cells = cove_cells
        self.linear_interpolation = linear_interpolation
        self.sess = sess

    def convert_spn_to_sparse_span_iterator(self):
        full_indices = tf.cast(tf.stack([tf.range(self.options.max_ctx_length)]*self.batch_size), dtype=tf.int64)
        firs_indices = (full_indices >= tf.expand_dims(self.spn_iterator[:,0], -1))
        second_indices = (full_indices <= tf.expand_dims(self.spn_iterator[:,1], -1))
        dense = tf.cast(firs_indices&second_indices, dtype=tf.int32)
        return convert_dense_to_sparse_tensor(dense), dense

    def get_use_dropout_placeholder(self):
        return self.use_dropout_placeholder

    def get_data_index_iterator(self):
        return self.data_index_iterator

    def get_keep_prob_placeholder(self):
        return self.keep_prob

    def get_input_keep_prob_placeholder(self):
        return self.input_keep_prob

    def get_rnn_keep_prob_placeholder(self):
        return self.rnn_keep_prob

    def setup(self):
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.use_dropout_placeholder = tf.placeholder(tf.bool, name="use_dropout")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
        self.rnn_keep_prob = tf.placeholder(tf.float32, name="rnn_keep_prob")
        self.batch_size = tf.shape(self.ctx_iterator)[0]
        self.ctx_len = tf.reduce_sum(tf.cast(tf.not_equal(self.ctx_iterator, self.sq_dataset.vocab.PAD_ID), tf.int32), axis=1)
        self.sparse_span_iterator, self.dense_span_iterator = self.convert_spn_to_sparse_span_iterator()
        model_inputs = create_model_inputs(self.sess,
                self.embeddings, self.ctx_iterator,
                self.qst_iterator,
                self.options, self.wiq_iterator,
                self.wic_iterator, self.sq_dataset,
                self.ctx_pos_iterator, self.qst_pos_iterator,
                self.ctx_ner_iterator, self.qst_ner_iterator,
                self.word_chars, self.cove_cells,
                self.use_dropout_placeholder,
                self.batch_size, self.input_keep_prob, self.keep_prob,
                self.rnn_keep_prob)
        self.ctx_inputs = model_inputs.ctx_concat
        self.qst_inputs = model_inputs.qst_concat
        self.ctx_glove = model_inputs.ctx_glove
        self.qst_glove = model_inputs.qst_glove
        self.ctx_cove = model_inputs.ctx_cove
        self.qst_cove = model_inputs.qst_cove

    def get_qst(self):
        return self.qst_iterator

    def get_start_spans(self):
        return tf.argmax(self.get_start_span_probs(), axis=1)

    def get_end_spans(self):
        return tf.argmax(self.get_end_span_probs(), axis=1)

    def get_loss_op(self):
        return self.loss

    def get_start_span_probs(self):
        return self.start_span_probs

    def get_end_span_probs(self):
        return self.end_span_probs
