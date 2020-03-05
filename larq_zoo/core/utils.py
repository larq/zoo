import contextlib
import json
import os
import sys
from typing import Optional

import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import (
    decode_predictions as keras_decode_predictions,
)
from tensorflow.python.eager.context import num_gpus
from tensorflow.python.keras.backend import is_keras_tensor


def slash_join(*args):
    return "/".join(arg.strip("/") for arg in args)


def download_pretrained_model(
    model: str,
    version: str,
    file: str,
    file_hash: str,
    cache_dir: Optional[str] = None,
) -> str:
    root_url = "https://github.com/larq/zoo/releases/download/"

    url = slash_join(root_url, model + "-" + version, file)
    cache_subdir = os.path.join("larq/models/", model)

    return keras.utils.get_file(
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
    except Exception:
        return 0


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
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
        data_format=keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )


def get_input_layer(input_shape, input_tensor):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape)
    if not is_keras_tensor(input_tensor):
        return keras.layers.Input(tensor=input_tensor, shape=input_shape)
    return input_tensor


def decode_predictions(preds, top=5, **kwargs):
    """Decodes the prediction of an ImageNet model.

    # Arguments
    preds: Numpy tensor encoding a batch of predictions.
    top: Integer, how many top-guesses to return.

    # Returns
    A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
    ValueError: In case of invalid shape of the `pred` array (must be 2D).
    """
    return keras_decode_predictions(preds, top=top, **kwargs)
