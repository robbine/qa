"""Provides SQuAD data for training and dev.
"""

import glob
import numpy as np
import os
import pickle
import preprocessing.constants as constants
import preprocessing.embedding_util as embedding_util
import re
import tensorflow as tf
import glob

from datasets.iterator_wrapper import *
from datasets.file_util import *
from preprocessing.vocab import get_vocab
from util.file_util import *


# Class that provides Squad data through Tensorflow iterators by cycling through
# a set of Numpy & pickle files. There doesn't seem to be a native way to do this
# easily through the Dataset API.
class SquadTFData:
    def __init__(self, options):
        self.options = options
        training_dir = os.path.join(options.data_dir,
            constants.TRAIN_FOLDER_NAME)
        validation_dir = os.path.join(options.data_dir,
            constants.DEV_FOLDER_NAME)
        self.vocab = get_vocab(options.data_dir)
        parser = get_record_parser(options)

        training_files = glob.glob(training_dir + '/*.tfrecords')
        validation_files = glob.glob(validation_dir + '/*.tfrecords')
        self.train_ds = get_batch_dataset(training_files, parser, options)
        self.dev_ds = get_dataset(validation_files, parser, options)


        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_ds.output_types,
            self.train_ds.output_shapes)
        self.train_iterator = self.train_ds.make_one_shot_iterator()
        self.dev_iterator = self.dev_ds.make_one_shot_iterator()

        self.embeddings = embedding_util.load_word_embeddings_including_unk_and_padding(options)
        self.word_chars = embedding_util.load_word_char_embeddings(options)

        self.word_vec_size = constants.WORD_VEC_DIM
        self.max_word_len = constants.MAX_WORD_LEN

        self.train_question_ids_to_squad_question_id = load_text_file(os.path.join(training_dir, constants.QUESTION_IDS_TO_SQUAD_QUESTION_ID))
        self.dev_question_ids_to_squad_question_id = load_text_file(os.path.join(validation_dir, constants.QUESTION_IDS_TO_SQUAD_QUESTION_ID))

        self.train_question_ids_to_passage_context = load_text_file(os.path.join(training_dir, constants.QUESTION_IDS_TO_PASSAGE_CONTEXT))
        self.dev_question_ids_to_passage_context = load_text_file(os.path.join(validation_dir, constants.QUESTION_IDS_TO_PASSAGE_CONTEXT))

        self.train_total = load_text_file(os.path.join(training_dir, "total"))
        self.dev_total = load_text_file(os.path.join(validation_dir, "total"))

    def get_max_ctx_len(self):
        return self.options.max_ctx_length

    def get_max_qst_len(self):
        return self.options.max_qst_length

    def get_word_vec_size(self):
        return self.word_vec_size

    def setup_with_tf_session(self, sess):
        print("Setting up tensorflow data iterator handles")
        self.train_handle = sess.run(self.train_iterator.string_handle())
        self.dev_handle = sess.run(self.dev_iterator.string_handle())

    def get_iterator_handle(self):
        return self.handle

    def get_train_handle(self):
        return self.train_handle

    def get_dev_handle(self):
        return self.dev_handle

    def get_sentences_for_all_gnd_truths(self, question_id, is_train):
        passage_context = self.train_question_ids_to_passage_context[question_id] if is_train else self.dev_question_ids_to_passage_context[question_id]
        return passage_context.acceptable_gnd_truths

    def get_sentence(self, question_id, start_idx, end_idx, is_train):
        # A 'PassageContext' defined in preprocessing/create_train_data.py
        passage_context = self.train_question_ids_to_passage_context[question_id] if is_train else self.dev_question_ids_to_passage_context[question_id]
        max_word_id = max(passage_context.word_id_to_text_positions.keys())
        text_start_idx = passage_context.word_id_to_text_positions[min(start_idx, max_word_id)].start_idx
        text_end_idx = passage_context.word_id_to_text_positions[min(end_idx, max_word_id)].end_idx
        return passage_context.passage_str[text_start_idx:text_end_idx]

    def question_ids_to_squad_ids(self, is_train):
        return self.train_question_ids_to_squad_question_id if is_train else self.dev_question_ids_to_squad_question_id

    def get_total(self, is_train):
        return self.train_total if is_train else self.dev_total
