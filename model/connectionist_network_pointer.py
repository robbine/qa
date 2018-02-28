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
        ctx_rs = tf.reshape(tf.transpose(ctx, [1,0,2]), [-1, ctx_dim])
        ctx_list = tf.split(ctx_rs, options.max_ctx_length, 0)
        n_hidden = options.rnn_size
        weigths_out_h1 = tf.Variable(tf.truncated_normal([2, n_hidden],
                                                   stddev=np.sqrt(2.0 / (2*n_hidden))))
        biases_out_h1 = tf.Variable(tf.zeros([n_hidden]))
        weights_classes = tf.Variable(tf.truncated_normal([n_hidden, 3],
                                                     stddev=np.sqrt(2.0 / n_hidden)))
        biases_classes = tf.Variable(tf.zeros([3]))
        ####Network
        forward_h1 = tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True, state_is_tuple=True)
        backward_h1 = tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True, state_is_tuple=True)
        fb_h1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(forward_h1, backward_h1, ctx_list, dtype=tf.float32,
                                                             scope='BDLSTM_H1')
        fb_h1rs = [tf.reshape(t, [batch_size, 2, n_hidden]) for t in fb_h1]
        out_h1 = [tf.reduce_sum(tf.multiply(t, weigths_out_h1), reduction_indices=1) + biases_out_h1 for t in fb_h1rs]

        logits = [tf.matmul(t, weights_classes) + biases_classes for t in out_h1]

        ####Optimizing
        logits3d = tf.stack(logits)
        loss = tf.reduce_mean(ctc.ctc_loss(sparse_span_iterator, logits3d, ctx_lens))

        ####Evaluating
        #predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, ctx_lens)[0][0])
        return loss
