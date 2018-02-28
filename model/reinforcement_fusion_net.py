"""Implements the "FusionNet" model.
https://arxiv.org/pdf/1711.07341.pdf
"""

import tensorflow as tf

from model.alignment import *
from model.base_model import BaseModel
from model.dropout_util import *
from model.encoding_util import *
from model.fusion_net_decoder import *
from model.fusion_net_util import *
from model.memory_answer_pointer import *
from model.rnn_util import *
from model.stochastic_answer_pointer import *
from model.connectionist_network_pointer import *

class ReinforcementFusionNet(BaseModel):
    def setup(self):
        super(ReinforcementFusionNet, self).setup()
        # Step 1. Form the "low-level" and "high-level" representations of
        # the context and question.
        ctx_low_level, ctx_high_level = \
            encode_low_level_and_high_level_representations(
                self.sess, "ctx_preprocessing", self.options, self.ctx_inputs,
                self.rnn_keep_prob, self.batch_size, self.use_dropout_placeholder)
        qst_low_level, qst_high_level = \
            encode_low_level_and_high_level_representations(
                self.sess, "qst_preprocessing", self.options, self.qst_inputs,
                self.rnn_keep_prob, self.batch_size, self.use_dropout_placeholder)

        # Step 2. Get the "question understanding" representation.
        qst_understanding = run_bidirectional_cudnn_lstm("qst_understanding",
            tf.concat([qst_low_level, qst_high_level], axis=-1),
            self.rnn_keep_prob, self.options, self.batch_size, self.sess,
            self.use_dropout_placeholder) # size = [batch_size, max_qst_length, 2 * rnn_size]

        # Step 3. Fuse the "history-of-word" question vectors into the
        # "history-of-word" context vectors.
        if self.options.use_cove_vectors:
            ctx_how = tf.concat([self.ctx_glove, self.ctx_cove,
                sequence_dropout(ctx_low_level, self.keep_prob),
                sequence_dropout(ctx_high_level, self.keep_prob)],
                axis=-1)
            qst_how = tf.concat([self.qst_glove, self.qst_cove,
                sequence_dropout(qst_low_level, self.keep_prob),
                sequence_dropout(qst_high_level, self.keep_prob)],
                axis=-1)
        else:
            ctx_how = tf.concat([self.ctx_glove,
                sequence_dropout(ctx_low_level, self.keep_prob),
                sequence_dropout(ctx_high_level, self.keep_prob)],
                axis=-1)
            qst_how = tf.concat([self.qst_glove,
                sequence_dropout(qst_low_level, self.keep_prob),
                sequence_dropout(qst_high_level, self.keep_prob)],
                axis=-1)
        how_dim = ctx_how.get_shape()[-1]
        ctx_low_fusion = vector_fusion("ctx_qst_low_fusion", self.options,
            ctx_how, qst_how, how_dim, qst_low_level, 1.0)
        ctx_high_fusion = vector_fusion("ctx_qst_high_fusion", self.options,
            ctx_how, qst_how, how_dim, qst_high_level, 1.0)
        ctx_understanding_fusion = vector_fusion("ctx_qst_understanding_fusion",
            self.options, ctx_how, qst_how, how_dim, qst_understanding,
            1.0)
        ctx_fusion_input = tf.concat([
            ctx_low_level, ctx_high_level, ctx_low_fusion, ctx_high_fusion,
            ctx_understanding_fusion], axis=-1)
        ctx_full_qst_fusion = run_bidirectional_cudnn_lstm("ctx_qst_fusion",
            ctx_fusion_input, self.rnn_keep_prob, self.options,
            self.batch_size, self.sess,
            self.use_dropout_placeholder) # size = [batch_size, max_ctx_length, 2 * rnn_size]

        # Step 4. Use the "history-of-word" context vectors to perform
        # self matching and then get the final context "understanding" vectors.
        self_matching_ctx_how = tf.concat([
            ctx_how,
            sequence_dropout(ctx_low_fusion, self.keep_prob),
            sequence_dropout(ctx_high_fusion, self.keep_prob),
            sequence_dropout(ctx_understanding_fusion, self.keep_prob),
            sequence_dropout(ctx_full_qst_fusion, self.keep_prob)], axis=-1)
        how_dim = self_matching_ctx_how.get_shape()[-1]
        self_matching_fusion = vector_fusion("self_matching_fusion",
             self.options, self_matching_ctx_how, self_matching_ctx_how,
             how_dim, ctx_full_qst_fusion, 1.0)
        final_ctx = run_bidirectional_cudnn_lstm("final_ctx",
            tf.concat([ctx_full_qst_fusion, self_matching_fusion], axis=-1),
            self.rnn_keep_prob, self.options, self.batch_size, self.sess,
            self.use_dropout_placeholder) # size = [batch_size, max_ctx_length, 2 * rnn_size]

        # Step 5. Decode the answer start & end.
        self.fusion_loss, self.start_span_probs, self.end_span_probs = \
            stochastic_answer_pointer(self.options, final_ctx, qst_understanding,
                self.spn_iterator, self.sq_dataset, self.keep_prob,
                self.sess, self.batch_size, self.use_dropout_placeholder)
# Alternative:
#        self.loss, self.start_span_probs, self.end_span_probs = \
#            decode_fusion_net(self.options, self.sq_dataset, self.keep_prob,
#                final_ctx, qst_understanding, self.batch_size, self.spn_iterator,
#                self.sess, self.use_dropout_placeholder)
        self.ctc_loss = connectionist_network_pointer(self.options, final_ctx, qst_understanding,
                self.sparse_span_iterator, self.sq_dataset, self.keep_prob,
                self.sess, self.batch_size, self.use_dropout_placeholder, self.ctx_len)
        self.loss = tf.cond(tf.equal(self.linear_interpolation, tf.constant(1.0)),
                                     lambda: self.fusion_loss,
                                     lambda: self.fusion_loss * self.linear_interpolation + self.ctc_loss * (1 - self.linear_interpolation))
