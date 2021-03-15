### `build_dataset`

```python
kt.image.build_dataset(paths: List[str], labels=None, bsize: int = 32, cache: bool = True, decode_fn=None, augment_fn=None, augment: bool = True, repeat: bool = True, shuffle: int = 1024, cache_dir: str = '')
```

*Build a tf.data.Dataset from a given list of path.*

- **paths** *(List[str]; required)*: The full (absolute or relative) paths of the files you want to load as inputs. This could be images or anything you want to preprocess.
- **labels** *(unspecified; optional)*: The target of your predictions. If left blank, the tf.data.Dataset will not output any label alongside your training examples.
- **bsize** *(int; default=32)*: The batch size.
- **cache** *(bool; default=True)*: Whether to cache the preprocessed images.
- **decode_fn** *(unspecified; optional)*: A custom function that will take as input the paths and output the tensors that will be given to the model.
- **augment_fn** *(unspecified; optional)*: A custom function that is applied to the decoded inputs before they are fed to the model.
- **augment** *(bool; default=True)*: Whether to apply the augment function
- **repeat** *(bool; default=True)*: Whether to repeat the dataset after one pass. This should be `True` if it is the training split, and `False` for test.
- **shuffle** *(int; default=1024)*: Number of examples to start shuffling. If set to N, then the first N examples from paths will be randomly shuffled.