import tensorflow as tf


def auto_select(verbose: bool=True) -> "tf.distribute.Strategy":
    """
    *Automatically select an accelerator depending on availability, and in the following order: TPU, GPU, CPU.*

    {{params}}
    {{verbose}} Whether to display which device was selected.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)

        if verbose:
            print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()

    if verbose:
        print(f"Running on {strategy.num_replicas_in_sync} replicas")

    return strategy
