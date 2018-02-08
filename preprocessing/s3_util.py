"""Uploads training data to AWS S3, if configured.
"""

import glob
import os
import time
import oss2
access_key_id = os.getenv('ossid')
access_key_secret = os.getenv('osskey')
endpoint = os.getenv('osshost')

def _everything_after_first_slash(s):
    pieces = s.split("/")
    if len(pieces) == 1:
        return s
    return "/".join(pieces[1:])

def _get_s3_files_in_bucket(options, bucket):
    key_prefix = options.s3_data_folder_name
    data_objs = {_everything_after_first_slash(obj.key) for obj in bucket.list_objects(prefix=key_prefix).object_list}
    return data_objs

def _get_existing_data_files(options):
    # Grab everything in the data directory. This is a scalable solution, but
    # it also assumes that no extraneous files are placed in the directory.
    data_files = glob.glob(os.path.join(options.data_dir, "**/*"))
    data_files = [f for f in data_files if not os.path.isdir(f)]
    data_files = [_everything_after_first_slash(f) for f in data_files] # Remove the `data_dir` prefix.
    return data_files

def _already_uploaded_s3_files(options, bucket, save_files):
    s3_files = _get_s3_files_in_bucket(options, bucket)
    return all([file_name in s3_files for file_name in save_files])

def maybe_upload_data_files_to_s3(options):
    """Uploads preprocessed data to S3 storage, if s3 is
       enabled and those files haven't already been uploaded.
    """
    if not options.use_s3:
        print("S3 not enabled; not uploading to S3.")
        return
    start = time.time()
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket(options.s3_bucket_name)
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, options.s3_bucket_name)
    data_files = _get_existing_data_files(options)
    if _already_uploaded_s3_files(options, bucket, data_files):
        print("Already uploaded all data files to s3. Not reuploading.")
        return
    print("Uploading data files to S3")
    for file_name in data_files:
        key = os.path.join(options.s3_data_folder_name, file_name)
        full_file_name = os.path.join(options.data_dir, file_name)
        print("Uploading %s to %s" % (full_file_name, key))
        # bucket.upload_file(full_file_name, key)
        bucket.put_object_from_file(key, full_file_name)
    print("Uploaded %d data files to AWS S3 in %f seconds" % (
                len(data_files), time.time() - start))

def _maybe_create_directories_for_file(options, s3_file):
    split_dirs = s3_file.split("/")
    dirs = "/".join(split_dirs[:-1])
    if dirs:
        try:
            os.makedirs(os.path.join(options.data_dir, dirs))
        except:
            print("exist")

def maybe_download_data_files_from_s3(options):
    """Downloads preprocessed training data from S3 storage, if s3 is
       enabled and those files haven't already been downloaded.
    """
    if not options.use_s3:
        print("S3 not enabled; not downloading from S3.")
        return
    if not os.path.isdir(options.data_dir):
        os.makedirs(options.data_dir)
    print("Downloading data files from S3")
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket(options.s3_bucket_name)
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, options.s3_bucket_name)
    data_files = _get_existing_data_files(options)
    s3_files = _get_s3_files_in_bucket(options, bucket)
    start = time.time()
    for s3_file in s3_files:
        _maybe_create_directories_for_file(options, s3_file)
        s3_key = os.path.join(options.s3_data_folder_name, s3_file)
        local_file_name = os.path.join(options.data_dir, s3_file)
        if os.path.exists(local_file_name):
            print("Already downloaded file %s. Not downloading again."
                  % local_file_name)
            continue
        bucket.get_object_to_file(s3_key, local_file_name)
    print("Time to download %d data files: %s" % (len(s3_files),
                time.time() - start))
