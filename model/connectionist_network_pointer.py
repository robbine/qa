"""The stochastic answer pointer from https://arxiv.org/pdf/1712.03556.pdf.
"""


import tensorflow as tf

from model.cudnn_lstm_wrapper import *
from model.dropout_util import *
from model.rnn_util import *
from model.tf_util import *
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc


def connectionist_network_pointer(options, ctx, qst, sparse_span_iterator, sq_dataset, keep_prob,
    sess, batch_size, use_dropout, ctx_lens):
    """Runs a stochastic answer pointer to get start/end span predictions

       Input:
         ctx: The passage representation of shape [batch_size, M, d]
         qst: The question representation of shape [batch_size, N, d]
         sparse_span_iterator: The target spans sparse tensor
         sq_dataset: A SquadDataBase object
         keep_prob: Probability used for dropout.

       Output:
         (loss, start_span_probs, end_span_probs)
         loss - a single scalar
         start_span_probs - the probabilities of the start spans of shape
            [batch_size, M]
         end_span_probs - the probabilities of the end spans of shape
            [batch_size, M]
    """
    with tf.variable_scope("connectionist_network_pointer"):
        max_qst_len = sq_dataset.get_max_qst_len()
        max_ctx_len = sq_dataset.get_max_ctx_len()
        ctx_dim = ctx.get_shape()[-1].value # 2 * rnn_size
        assert qst.get_shape()[-1].value == ctx.get_shape()[-1].value
        w = tf.get_variable("w", shape=[ctx_dim], dtype=tf.float32)
        Qw = multiply_tensors(qst, w) # size = [batch_size, max_qst_length]
        sm = tf.nn.softmax(Qw, dim=1) # size = [batch_size, max_qst_length]
        s = tf.matmul(tf.reshape(sm, [batch_size, 1, max_qst_len])
            , qst) # size = [batch_size, 1, 2 * rnn_size]

        lstm = create_cudnn_lstm(ctx_dim,
            sess, options, "lstm", keep_prob,
            bidirectional=False, layer_size=ctx_dim, num_layers=1)
        state_h = s
        state_c = s

        weights_classes = tf.get_variable("weigth", shape=[ctx_dim, 3], dtype=tf.float32)
        biases_classes = tf.get_variable("bias", shape=[3], dtype=tf.float32)
        s, state_h, state_c = run_cudnn_lstm(ctx, keep_prob, options,
            lstm, batch_size, use_dropout,
            initial_state_h=state_h, initial_state_c=state_c) # size(s) = [batch_size, max_ctx_len, rnn_size]
        fb_h1rs = [tf.squeeze(t) for t in tf.split(s, max_ctx_len, axis=1)] # size(fb_h1rs) = [batch_size, rnn_size]
        logits = [tf.matmul(t, weights_classes) + biases_classes for t in fb_h1rs]
        logits3d = tf.stack(logits) + options.numerical_stability_epsilon
        loss = tf.reduce_mean(ctc.ctc_loss(sparse_span_iterator, logits3d, ctx_lens, ctc_merge_repeated=False))

        ####Evaluating
        #predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, ctx_lens)[0][0])
        return loss
