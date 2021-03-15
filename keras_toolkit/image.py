import os
from typing import List

import tensorflow as tf


def build_decoder(with_labels=True, target_size=(256, 256), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)

        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths: List[str],
    labels=None,
    bsize: int = 32,
    cache: bool = True,
    decode_fn=None,
    augment_fn=None,
    augment: bool = True,
    repeat: bool = True,
    shuffle: int = 1024,
    cache_dir: str = "",
):
    """
    *Build a tf.data.Dataset from a given list of path.*

    {{ params }}
    {{paths}} The full (absolute or relative) paths of the files you want to load as inputs. This could be images or anything you want to preprocess.
    {{labels}} The target of your predictions. If left blank, the tf.data.Dataset will not output any label alongside your training examples.
    {{bsize}} The batch size.
    {{cache}} Whether to cache the preprocessed images.
    {{decode_fn}} A custom function that will take as input the paths and output the tensors that will be given to the model.
    {{augment_fn}} A custom function that is applied to the decoded inputs before they are fed to the model.
    {{augment}} Whether to apply the augment function
    {{repeat}} Whether to repeat the dataset after one pass. This should be `True` if it is the training split, and `False` for test.
    {{shuffle}} Number of examples to start shuffling. If set to N, then the first N examples from paths will be randomly shuffled.
    """
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)

    return dset