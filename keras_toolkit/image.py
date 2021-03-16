import os
from typing import List, Any, Optional, Callable, Tuple, Union, TypeVar

import tensorflow as tf

Label = TypeVar('Label')


def build_decoder(
    with_labels: bool = True, target_size: Tuple[int, int] = (256, 256), ext: str = "jpg"
) -> Callable:
    """
    _Build a decoder function that will be called by `tf.data.Dataset` every time it wants to
    fetch an image to add to the next batch. The decoder function will process one sample 
    at the time and returns a `tf.Tensor` of type `float` and shape `(b, *target_size, 3)`,
    where `b` is the batch size that will be specified when calling build_dataset._

    {{params}}
    {{with_labels}} Whether the decoder will receive as input a label and output that same label without any change.
    {{target_size}} A 2-tuple indicating the height and the width of the resized image (resizing is automatically performed).
    {{ext}} The extension of the image file. Must be either "png" or "jpg"; no other format is currently supported.

    **Note**:
    - If `with_labels` is `True`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`
    - If `with_labels` is `False`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`
    """

    def decode(path: str) -> tf.Tensor:
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

    def decode_with_labels(path: str, label: Label) -> Tuple[tf.Tensor, Label]:
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True) -> Callable:
    """
    _Build an augment function that will randomly flip the input image left-right and up-and-down._

    {{params}}
    {{with_labels}} Whether the decoder will receive as input a label and output that same label without any change.

    **Note**:
    - If `with_labels` is `True`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`
    - If `with_labels` is `False`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`
    """
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths: List[str],
    labels: Optional[Any] = None,
    bsize: int = 32,
    cache: bool = True,
    decode_fn: Callable = None,
    augment_fn: Callable = None,
    augment: bool = True,
    repeat: bool = True,
    shuffle: int = 1024,
    cache_dir: str = "",
) -> "tf.data.Dataset":
    """
    *Build a tf.data.Dataset from a given list of paths, and optionally labels. This dataset can be used to fit a Keras model, i.e. `model.fit(data)` where `data=build_dataset(...)`*

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