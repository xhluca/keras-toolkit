# API Reference


## kt.accelerator


### auto_select

```python
kt.accelerator.auto_select(verbose: bool = True) -> 'tf.distribute.Strategy'
```

*Automatically select an accelerator depending on availability, and in the following order: TPU, GPU, CPU.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **verbose** | *bool* | `True` | Whether to display which device was selected.

<br>


### limit_gpu_memory

```python
kt.accelerator.limit_gpu_memory(limit: str = None, verbose: bool = True, device: int = 0)
```

*Limits the amount of GPU memory (VRAM) used by TensorFlow.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **limit** | *str* | *optional* | The limit value in megabytes.
| **verbose** | *bool* | `True` | Whether to display the restriction message.
| **device** | *int* | `0` | Which GPU we want to limit the memory, if there are more than one.

**Warning: This function uses experimental features from TensorFlow!**

<br>



<br>


## kt.image


### build_decoder

```python
kt.image.build_decoder(with_labels: bool = True, target_size: Tuple[int, int] = (256, 256), ext: str = 'jpg') -> Callable
```

_Build a decoder function that will be called by `tf.data.Dataset` every time it wants to
fetch an image to add to the next batch. The decoder function will process one sample
at the time and returns a `tf.Tensor` of type `float` and shape `(b, *target_size, 3)`,
where `b` is the batch size that will be specified when calling build_dataset._

| Parameter | Type | Default | Description |
|-|-|-|-|
| **with_labels** | *bool* | `True` | Whether the decoder will receive as input a label and output that same label without any change.
| **target_size** | *Tuple[int, int]* | `(256, 256)` | A 2-tuple indicating the height and the width of the resized image (resizing is automatically performed).
| **ext** | *str* | `jpg` | The extension of the image file. Must be either "png" or "jpg"; no other format is currently supported.

**Note**:
- If `with_labels` is `True`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`
- If `with_labels` is `False`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`

<br>


### build_augmenter

```python
kt.image.build_augmenter(with_labels=True) -> Callable
```

_Build an augment function that will randomly flip the input image left-right and up-and-down._

| Parameter | Type | Default | Description |
|-|-|-|-|
| **with_labels** | *unspecified* | `True` | Whether the decoder will receive as input a label and output that same label without any change.

**Note**:
- If `with_labels` is `True`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`
- If `with_labels` is `False`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`

<br>


### build_dataset

```python
kt.image.build_dataset(paths: List[str], labels: Union[Any, NoneType] = None, decode_fn: Callable = None, bsize: int = 32, cache: Union[bool, str] = False, augment: Union[bool, Callable] = False, repeat: bool = False, shuffle: int = 1024, random_state: int = None) -> 'tf.data.Dataset'
```

*Build a tf.data.Dataset from a given list of paths, and optionally labels. This dataset can be used to fit a Keras model*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **paths** | *List[str]* | *required* | The full (absolute or relative) paths of the files you want to load as inputs. This could be images or anything you want to preprocess.
| **labels** | *optional* | *optional* | The target of your predictions. If left blank, the tf.data.Dataset will not output any label alongside your training examples.
| **decode_fn** | *Callable* | *optional* | A custom function that will take as input the paths and output the tensors that will be given to the model.
| **bsize** | *int* | `32` | The batch size, i.e. the number of examples processed at once.
| **cache** | *Union[bool, str]* | `False` | This can be a boolean (`True` for in-memory caching, `False` for no caching) or a string value representing a path.
| **augment** | *Union[bool, Callable]* | `False` | This can be a boolean indicating whether to apply default augmentations, or a function that will be applied to the decoded inputs before they are fed to the model.
| **repeat** | *bool* | `False` | Whether to repeat the dataset after one pass. This should be `True` if it is the training split, and `False` for test.
| **shuffle** | *int* | `1024` | Number of examples to start shuffling, corresponding to the buffer size.
| **random_state** | *int* | *optional* | An integer representing the random seed that will be used to create the distribution.


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

<br>



<br>

