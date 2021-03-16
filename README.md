# Keras Toolkit

*A collection of functions to help you easily train and run Tensorflow Keras*

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
dtrain = kt.data.build_dataset(paths, labels, bsize=BATCH_SIZE, decode_fn=decoder)

with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile(...)

model.fit(...)
```

### API References

Get the complete [API reference here](https://github.com/xhlulu/keras-toolkit/blob/master/docs/REFERENCES.md).


## Acknowledgement

The `auto_select_accelerator` was inspired by [Martin Gorner's Kaggle notebook](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu).