import os
from typing import List, Any, Optional, Callable, Tuple, Union, TypeVar
import inspect

import tensorflow as tf

Label = TypeVar("Label")


def build_decoder(
    with_labels: bool = True,
    target_size: Tuple[int, int] = (256, 256),
    ext: str = "jpg",
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
    decode_fn: Callable = None,
    bsize: int = 32,
    cache: Union[bool, str] = False,
    augment: Union[bool, Callable] = False,
    repeat: bool = False,
    shuffle: int = 1024,
    random_state: int = None
) -> "tf.data.Dataset":
    """
    *Build a tf.data.Dataset from a given list of paths, and optionally labels. This dataset can be used to fit a Keras model*

    {{ params }}
    {{paths}} The full (absolute or relative) paths of the files you want to load as inputs. This could be images or anything you want to preprocess.
    {{labels}} The target of your predictions. If left blank, the tf.data.Dataset will not output any label alongside your training examples.
    {{decode_fn}} A custom function that will take as input the paths and output the tensors that will be given to the model.
    {{bsize}} The batch size, i.e. the number of examples processed at once.
    {{cache}} This can be a boolean (`True` for in-memory caching, `False` for no caching) or a string value representing a path.
    {{augment}} This can be a boolean indicating whether to apply default augmentations, or a function that will be applied to the decoded inputs before they are fed to the model.
    {{repeat}} Whether to repeat the dataset after one pass. This should be `True` if it is the training split, and `False` for test.
    {{shuffle}} Number of examples to start shuffling, corresponding to the buffer size.
    {{random_state}} An integer representing the random seed that will be used to create the distribution.

    
    ### Notes
    
    - If set to N, then initially the first N examples from `paths` will be randomly shuffled, and after every batch processed the subsequent paths will be added such that there are always N examples to choose from.

    ### Example

    ```python
    paths = ["./train/image1.png", "./train/image2.png", "./train/image3.png"]
    labels = [0, 1, 0]
    
    dtrain = build_dataset(paths, labels)
    
    model = tf.keras.Sequential([...])
    model.fit(dtrain, epochs=1)
    ```
    """
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    # Apply caching
    if cache is True:
        dset = dset.cache()
    elif type(cache) is str:
        os.makedirs(cache, exist_ok=True)
        dset = dset.cache(cache)
    elif cache is False:
        pass
    else:
        raise ValueError("Invalid 'cache' argument. Please choose a boolean or a string.")


    # Apply augmentation
    if augment is True:
        augment_fn = build_augmenter(labels is not None)
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)
    elif inspect.isfunction(augment):
        dset = dset.map(augment, num_parallel_calls=AUTO)
    elif augment is False:
        pass
    else:
        raise ValueError("Invalid 'augment' argment. Please choose a boolean or a function.")
    
    # Apply repeat
    dset = dset.repeat() if repeat else dset

    # Apply shuffle
    if type(shuffle) is int:
        dset = dset.shuffle(shuffle, seed=random_state)
    elif shuffle is True:
        dset = dset.shuffle(shuffle, seed=random_state)
        
    # Apply batching
    dset = dset.batch(bsize).prefetch(AUTO)

    return dset
