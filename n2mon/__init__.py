import hashlib

import tensorflow as tf

def md5(fname):
    """
    Get md5 of a downloaded file.
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_image(file_path):
    """
    Load a jpeg image at `file_path` into a tensor.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img