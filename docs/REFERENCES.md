
## `kt.accelerator`


### `auto_select`

```python
kt.accelerator.auto_select(verbose: bool = True) -> tensorflow.python.distribute.distribute_lib.Strategy
```

*Automatically select an accelerator depending on availability, and in the following order: TPU, GPU, CPU.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **verbose** | *bool* | `True` | Whether to display which device was selected.




## `kt.image`


### `build_dataset`

```python
kt.image.build_dataset(paths: List[str], labels: Union[Any, NoneType] = None, bsize: int = 32, cache: bool = True, decode_fn: Callable = None, augment_fn: Callable = None, augment: bool = True, repeat: bool = True, shuffle: int = 1024, cache_dir: str = '') -> tensorflow.python.data.ops.dataset_ops.DatasetV2
```

*Build a tf.data.Dataset from a given list of paths, and optionally labels. This dataset can be used to fit a Keras model, i.e. `model.fit(data)` where `data=build_dataset(...)*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **paths** | *List[str]* | *required* | The full (absolute or relative) paths of the files you want to load as inputs. This could be images or anything you want to preprocess.
| **labels** | *optional* | *optional* | The target of your predictions. If left blank, the tf.data.Dataset will not output any label alongside your training examples.
| **bsize** | *int* | `32` | The batch size.
| **cache** | *bool* | `True` | Whether to cache the preprocessed images.
| **decode_fn** | *Callable* | *optional* | A custom function that will take as input the paths and output the tensors that will be given to the model.
| **augment_fn** | *Callable* | *optional* | A custom function that is applied to the decoded inputs before they are fed to the model.
| **augment** | *bool* | `True` | Whether to apply the augment function
| **repeat** | *bool* | `True` | Whether to repeat the dataset after one pass. This should be `True` if it is the training split, and `False` for test.
| **shuffle** | *int* | `1024` | Number of examples to start shuffling. If set to N, then the first N examples from paths will be randomly shuffled.



### `build_augmenter`

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



### `build_decoder`

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
| **target_size** | *Tuple[int, int]* | `(256, 256)` | A 2-tuple indicating the height and the width of the resized image (resizing is automatically performend).
| **ext** | *str* | `jpg` | The extension of the image file. Must be either "png" or "jpg"; no other format is currently supported.

**Note**:
- If `with_labels` is `True`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`
- If `with_labels` is `False`, then the output function will have this signature: `decode(path: str) -> tf.Tensor`



