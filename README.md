# Keras Toolkit

*A collection of functions to help you easily train and run Tensorflow Keras*

Get the complete [API reference here](https://github.com/xhlulu/keras-toolkit/blob/master/docs/references.md).

## Quickstart

Install the library:

```
pip install keras-toolkit
```

You can now use it:
```python
import keras_toolkit as kt

# kt reduces the number of lines from ~100 to ~3
strategy = kt.accelerator.auto_select(verbose=True)
decoder = kt.image.build_decoder(with_labels=True, target_size=(300, 300))
dtrain = kt.image.build_dataset(paths, labels, bsize=BATCH_SIZE, decode_fn=decoder)

with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile(...)

model.fit(...)
```

## Usage

To automatically select an accelerator (e.g. TPU, GPU, CPU) and run on that accelerator:
```python
import keras_toolkit as kt
strategy = kt.accelerator.auto_select(verbose=True)

with strategy.scope():
    # your keras code here
    model = tf.keras.Sequential([...])
```

To restrict the GPU memory usage of TensorFlow (e.g. to 2GB):
```python
import keras_toolkit as kt

kt.accelerator.limit_gpu_memory(2*1024)
```

To build an image dataset from a list of paths and a list of labels (associated with the paths):
```python
import keras_toolkit as kt

dtrain = kt.image.build_dataset(paths, labels)
# => <PrefetchDataset shapes: ((None, 256, 256, 3), (None,)), types: (tf.float32, tf.int32)>

# Fit your keras model on that new tf.data.Dataset:
model.fit(dtrain, ...)
```

If you only have a list of image paths, it will create `tf.data.Dataset` without labels:
```python
dtrain = kt.image.build_dataset(paths)
# => <PrefetchDataset shapes: (None, 256, 256, 3), types: tf.float32>
```

You can also customize the dataset (e.g. batch size, custom image loader, custom augmentation, etc.):

```python
# This is just the default
img_decoder = kt.image.build_decoder(target_size=(512, 512))
augmenter = kt.image.build_augmenter()

dset = build_dataset(
    paths, labels, 
    decode_fn=img_decoder,
    bsize=64,
    cache="./cache_dir/",
    augment=augmenter,
    shuffle=False,
    random_state=42
)
```

## Acknowledgement

* The `kt.accelerator.auto_select_accelerator` was inspired by [Martin Gorner's Kaggle notebook](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu).
* The `kt.accelerator.limit_gpu_memory` was taken from [Chris Deotte's Kaggle notebook](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700).