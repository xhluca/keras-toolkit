import tensorflow as tf


def auto_select(verbose: bool = True) -> "tf.distribute.Strategy":
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


def limit_gpu_memory(limit: str = None, verbose: bool = True, device: int = 0):
    """
    *Limits the amount of GPU memory (VRAM) used by TensorFlow.*

    {{params}}
    {{limit}} The limit value in megabytes.
    {{verbose}} Whether to display the restriction message.
    {{device}} Which GPU we want to limit the memory, if there are more than one.

    **Warning: This function uses experimental features from TensorFlow!**
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[device],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)],
            )
        except RuntimeError as e:
            print(e)
        print(f"We will restrict TensorFlow to max {limit/1024:.2f}GB GPU RAM")
