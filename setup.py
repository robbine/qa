import os
import tensorflow as tf

from preprocessing.save_cove_weights import save_cove_weights
from preprocessing.create_train_data import DataParser
from preprocessing.download_data import download_data
from preprocessing.embedding_util import split_vocab_and_embedding
from preprocessing.s3_util import maybe_upload_data_files_to_s3
from flags import get_options_from_flags

def main(_):
    options = get_options_from_flags()
    data_dir = options.data_dir
    download_dir = options.download_dir
    for d in [data_dir, download_dir]:
        try:
            os.makedirs(d)
        except:
            print("exist")
    download_data(download_dir)
    split_vocab_and_embedding(data_dir, download_dir)
    DataParser(data_dir, download_dir).create_train_data()
    if options.use_cove_vectors:
        save_cove_weights(options)
    maybe_upload_data_files_to_s3(options)

if __name__ == "__main__":
    tf.app.run()
