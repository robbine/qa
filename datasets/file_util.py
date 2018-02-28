"""Functions for loading data from files.
"""

import numpy as np
import os
import pickle
import tensorflow as tf
import re
from collections import Counter
import string

def load_text_file(full_file_name):
    f = open(full_file_name, "rb")
    text_tokens = pickle.load(f)
    f.close()
    return text_tokens


def get_record_parser(options):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               'ctx_vocab_ids': tf.FixedLenFeature([], tf.string),
                                               'qst_vocab_ids': tf.FixedLenFeature([], tf.string),
                                               'ctx_pos_ids': tf.FixedLenFeature([], tf.string),
                                               'ctx_ner_ids': tf.FixedLenFeature([], tf.string),
                                               'span': tf.FixedLenFeature([2], tf.int64),
                                               'question_id': tf.FixedLenFeature([1], tf.int64),
                                               'qst_pos_ids': tf.FixedLenFeature([], tf.string),
                                               'qst_ner_ids': tf.FixedLenFeature([], tf.string),
                                               'ctx_in_qst': tf.FixedLenFeature([], tf.string),
                                               'qst_in_ctx': tf.FixedLenFeature([], tf.string),
                                           })

        ctx_vocab_ids = tf.reshape(tf.decode_raw(features["ctx_vocab_ids"], tf.int64), [options.max_ctx_length])
        qst_vocab_ids = tf.reshape(tf.decode_raw(features["qst_vocab_ids"], tf.int64), [options.max_qst_length])
        ctx_pos_ids = tf.reshape(tf.decode_raw(features["ctx_pos_ids"], tf.int64), [options.max_ctx_length])
        ctx_ner_ids = tf.reshape(tf.decode_raw(features["ctx_ner_ids"], tf.int64), [options.max_ctx_length])
        span = features["span"]
        question_id = features['question_id']
        qst_pos_ids = tf.reshape(tf.decode_raw(features['qst_pos_ids'], tf.int64), [options.max_qst_length])
        qst_ner_ids = tf.reshape(tf.decode_raw(features['qst_ner_ids'], tf.int64), [options.max_qst_length])
        ctx_in_qst = tf.reshape(tf.decode_raw(features['ctx_in_qst'], tf.int64), [options.max_ctx_length])
        qst_in_ctx = tf.reshape(tf.decode_raw(features['qst_in_ctx'], tf.int64), [options.max_qst_length])
        return ctx_vocab_ids, qst_vocab_ids, ctx_pos_ids, ctx_ner_ids, span, question_id, qst_pos_ids, qst_ner_ids, ctx_in_qst, qst_in_ctx
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(ctx_vocab_ids, qst_vocab_ids, ctx_pos_ids, ctx_ner_ids, span, question_id, qst_pos_ids, qst_ner_ids, ctx_in_qst, qst_in_ctx):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(ctx_vocab_ids, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset
