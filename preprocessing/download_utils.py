import os
import sys
import urllib2
import zipfile


def download_file_with_progress(url, filename):
    def display_progress(current_blocks, block_size_bytes, total_size):
        sys.stdout.write("Download progress: %d%%   \r" % int(current_blocks * block_size_bytes * 100.0 / total_size) )
        sys.stdout.flush()
    print("Downloading file", filename, "from url", url)
    with open(filename, 'wb') as f:
        f.write(urllib2.urlopen(url).read())
        f.close()

    print("")

def unzip_file_and_remove(zip_file_name, unzip_dir):
    print("Unzipping file", zip_file_name)
    zip_ref = zipfile.ZipFile(zip_file_name, "r")
    zip_ref.extractall(unzip_dir)
    zip_ref.close()
    os.remove(zip_file_name)
    print("Finished unzipping file", zip_file_name)
