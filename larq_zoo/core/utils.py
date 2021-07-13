import contextlib
import json
import os
import sys
from typing import Optional, Union

import tensorflow as tf
from tensorflow.python.eager.context import num_gpus

if tf.__version__[0] > "1":
    is_keras_tensor = tf.keras.backend.is_keras_tensor
else:
    from tensorflow.python.keras.backend import is_keras_tensor

try:
    from tensorflow.python.keras.applications.imagenet_utils import obtain_input_shape
except ImportError:
    # type: ignore
    from keras_applications.imagenet_utils import (
        _obtain_input_shape as obtain_input_shape,
    )

try:
    if tf.__version__[:3] > "2.5":
        from keras.engine.keras_tensor import KerasTensor
    else:
        from tensorflow.python.keras.engine.keras_tensor import KerasTensor

    # Necessary due to https://github.com/tensorflow/tensorflow/issues/44613
    TensorType = Union[tf.Tensor, KerasTensor]
except ImportError:
    TensorType = tf.Tensor


def slash_join(*args):
    return "/".join(arg.strip("/") for arg in args)


def download_pretrained_model(
    model: str, version: str, file: str, file_hash: str, cache_dir: Optional[str] = None
) -> str:
    root_url = "https://github.com/larq/zoo/releases/download/"

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
    except Exception:
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
    return obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=0,
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


def global_pool(
    x: tf.Tensor, data_format: str = "channels_last", name: str = None
) -> tf.Tensor:
    """Global average 2D pooling and flattening.

    Alternative to existing Keras implementation of GlobalAveragePooling2D.
    AveragePooling2D is much faster than GlobalAveragePooling2D on Larq Compute Engine.
    If the width or height of the input is dynamic, the function falls back to
    GlobalAveragePooling2D.

    # Arguments
        x: 4D TensorFlow tensor.
        data_format: A string, one of channels_last (default) or
            channels_first. The ordering of the dimensions in the inputs. channels_last
            corresponds to inputs with shape (batch, height, width, channels) while
            channels_first corresponds to inputs with shape (batch, channels, height,
            width). It defaults to "channels_last".
        name: String name of the layer

    # Returns
        2D TensorFlow tensor.

    # Raises
        ValueError: if tensor is not 4D or data_format is not recognized.
    """
    if len(x.get_shape()) != 4:
        raise ValueError("Tensor is not 4D.")
    if data_format not in ["channels_last", "channels_first"]:
        raise ValueError("data_format not recognized.")

    try:
        input_shape = x.get_shape().as_list()
        pool_size = (
            input_shape[1:3] if data_format == "channels_last" else input_shape[2:4]
        )
        x = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size,
            data_format=data_format,
            name=f"{name}_pool" if name else None,
        )(x)
        x = tf.keras.layers.Flatten(name=f"{name}_flatten" if name else None)(x)
    except ValueError:
        x = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format, name=name)(
            x
        )

    return x


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
    return tf.keras.applications.vgg16.decode_predictions(preds, top=top, **kwargs)


def TFOpLayer(tf_op: tf.Operation, *args, **kwargs) -> tf.keras.layers.Layer:
    """Wrap a tensorflow op using a Lambda layer. This facilitates naming the op as a
    proper keras layer.

    Example: `TFOpLayer(tf.split, groups, axis=-1, name="split")(x)`.

    # Arguments
        tf_op: tensorflow that needs to be wrapped.

    # Returns
        A keras layer wrapping `tf_op`.
    """
    name = kwargs.pop("name", None)
    return tf.keras.layers.Lambda(lambda x_: tf_op(x_, *args, **kwargs), name=name)
