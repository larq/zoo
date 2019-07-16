import os
import sys
import tensorflow as tf
import contextlib
import json
from tensorflow.python.keras.backend import is_keras_tensor
from tensorflow.python.eager.context import num_gpus
from keras_applications.imagenet_utils import _obtain_input_shape
from collections import namedtuple

ImagenetDataset = namedtuple("ImagenetDataset", ["input_shape", "num_classes"])


def slash_join(*args):
    return "/".join(arg.strip("/") for arg in args)


def download_pretrained_model(model, version, file, file_hash, cache_dir=None):
    root_url = "https://github.com/plumerai/larq-zoo/releases/download/"

    url = slash_join(root_url, model + "-" + version, file)
    cache_subdir = os.path.join("larq/models/", model)

    return tf.keras.utils.get_file(
        fname=file,
        origin=url,
        cache_dir=cache_dir,
        cache_subdir=cache_subdir,
        file_hash=file_hash,
    )


def get_current_epoch(output_dir):
    try:
        with open(os.path.join(output_dir, "stats.json"), "r") as f:
            return json.load(f)["epoch"]
    except:
        return 0


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with open(os.path.join(os.path.dirname(self.filepath), "stats.json"), "w") as f:
            return json.dump({"epoch": epoch + 1}, f)


def get_distribution_scope(batch_size):
    if num_gpus() > 1:
        strategy = tf.distribute.MirroredStrategy()
        assert (
            batch_size % strategy.num_replicas_in_sync == 0
        ), f"Batch size {batch_size} cannot be divided onto {num_gpus()} GPUs"
        distribution_scope = strategy.scope
    else:
        if sys.version_info >= (3, 7):
            distribution_scope = contextlib.nullcontext
        else:
            distribution_scope = contextlib.suppress

    return distribution_scope()


def validate_input(input_shape, weights, include_top, classes):
    if not (weights in {"imagenet", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either `None` (random initialization), "
            "`imagenet` (pre-training on ImageNet), or the path to the weights file "
            "to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            "If using `weights` as `imagenet` with `include_top` as true, "
            "`classes` should be 1000"
        )

    # Determine proper input shape
    return _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=64,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )


def get_input_layer(input_shape, input_tensor):
    if input_tensor is None:
        return tf.keras.layers.Input(shape=input_shape)
    if not is_keras_tensor(input_tensor):
        return tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
    return input_tensor
