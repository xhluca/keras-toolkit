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


## Acknowledgement

* The `kt.accelerator.auto_select_accelerator` was inspired by [Martin Gorner's Kaggle notebook](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu).
* The `kt.accelerator.limit_gpu_memory` was taken from [Chris Deotte's Kaggle notebook](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700).