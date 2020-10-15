from typing import Optional, Sequence

import larq as lq
import numpy as np
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


def blurpool_initializer(shape, dtype=None):
    ksize, filters = shape[0], shape[2]

    if ksize == 2:
        k = np.array([1, 1])
    elif ksize == 3:
        k = np.array([1, 2, 1])
    elif ksize == 5:
        k = np.array([1, 4, 6, 4, 1])
    else:
        raise ValueError("filter size should be in 2, 3, 5")

    k = np.outer(k, k)
    k = k / np.sum(k)
    k = np.expand_dims(k, axis=-1)
    k = np.repeat(k, filters, axis=-1)
    return np.reshape(k, shape)


@factory
class QuickNetFactory(ModelFactory):
    name = "quicknet"
    section_blocks: Sequence[int] = Field((4, 4, 4, 4))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0.0",
            file="quicknet_weights.h5",
            file_hash="7b4fa94f5241c7aad3412ca42b5db6517dbc4847cff710cb82be10c2f83bc0be",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v1.0.0",
            file="quicknet_weights_notop.h5",
            file_hash="359eed6dae43525eddf520ea87ec9b54750ee0e022647775d115a38856be396f",
        )

    @property
    def input_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_constraint(self):
        return lq.constraints.WeightClip(clip_value=1.25)

    def __post_configure__(self):
        assert len(self.section_blocks) == len(self.section_filters)

    def stem_module(self, filters: int, x: tf.Tensor) -> tf.Tensor:
        """Start of network."""
        assert filters % 4 == 0

        x = lq.layers.QuantConv2D(
            filters // 4,
            (3, 3),
            kernel_initializer="he_normal",
            padding="same",
            strides=2,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = lq.layers.QuantDepthwiseConv2D(
            (3, 3),
            padding="same",
            strides=2,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(scale=False, center=False)(x)

        x = lq.layers.QuantConv2D(
            filters,
            1,
            kernel_initializer="he_normal",
            use_bias=False,
        )(x)
        return tf.keras.layers.BatchNormalization()(x)

    def conv_block(
        self,
        x: tf.Tensor,
        filters: int,
        strides: int = 1,
    ) -> tf.Tensor:
        x = lq.layers.QuantConv2D(
            filters,
            (3, 3),
            activation="relu",
            input_quantizer=self.input_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer="glorot_normal",
            padding="same",
            pad_values=1.0,
            strides=strides,
            use_bias=False,
        )(x)
        return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    def residual_block(self, x: tf.Tensor) -> tf.Tensor:
        """Standard residual block, without strides or filter changes."""

        residual = x
        x = lq.layers.QuantConv2D(
            int(x.shape[-1]),
            (3, 3),
            activation="relu",
            input_quantizer=self.input_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer="glorot_normal",
            padding="same",
            pad_values=1.0,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        return x + residual

    def transition_block(
        self,
        x: tf.Tensor,
        filters: int,
        strides: int,
    ) -> tf.Tensor:
        """Pointwise transition block."""

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool2D(pool_size=strides, strides=1)(x)
        x = tf.keras.layers.DepthwiseConv2D(
            (3, 3),
            depthwise_initializer=blurpool_initializer,
            padding="same",
            strides=strides,
            trainable=False,
            use_bias=False,
        )(x)

        x = lq.layers.QuantConv2D(
            filters,
            (1, 1),
            kernel_initializer="glorot_normal",
            use_bias=False,
        )(x)
        return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    def build(self) -> tf.keras.models.Model:
        x = self.stem_module(self.section_filters[0], self.image_input)

        for block, (layers, filters) in enumerate(
            zip(self.section_blocks, self.section_filters)
        ):
            for layer in range(layers):
                if filters != x.shape[-1]:
                    strides = 1 if (block == 0 or layer != 0) else 2
                    x = self.transition_block(x, filters, strides)
                x = self.residual_block(x)

        if self.include_top:
            x = tf.keras.layers.Activation("relu")(x)
            x = utils.global_pool(x)
            x = lq.layers.QuantDense(
                self.num_classes,
                kernel_initializer="glorot_normal",
            )(x)
            x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=self.image_input, outputs=x, name=self.name)

        # Load weights.
        if self.weights == "imagenet":
            weights_path = (
                self.imagenet_weights_path
                if self.include_top
                else self.imagenet_no_top_weights_path
            )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


@factory
class QuickNetSmallFactory(QuickNetFactory):
    name = "quicknet_small"
    section_filters = Field((32, 64, 256, 512))

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_small",
            version="v1.0.0",
            file="quicknet_small_weights.h5",
            file_hash="6bf778e243466c678d6da0e3a91c77deec4832460046fca9e6ac8ae97a41299c",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_small",
            version="v1.0.0",
            file="quicknet_small_weights_notop.h5",
            file_hash="b65d59dd2d5af63d019997b05faff9e003510e2512aa973ee05eb1b82b8792a9",
        )


@factory
class QuickNetLargeFactory(QuickNetFactory):
    name = "quicknet_large"
    section_blocks = Field((6, 8, 12, 6))

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_large",
            version="v1.0.0",
            file="quicknet_large_weights.h5",
            file_hash="6bf778e243466c678d6da0e3a91c77deec4832460046fca9e6ac8ae97a41299c",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_large",
            version="v1.0.0",
            file="quicknet_large_weights_notop.h5",
            file_hash="b65d59dd2d5af63d019997b05faff9e003510e2512aa973ee05eb1b82b8792a9",
        )


def QuickNet(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the QuickNet architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    quicknet-v1.0.0/quicknet.json
    ```
    ```summary
    sota.QuickNet
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 63.3 %         | 84.6 %         | 13 234 088 | 4.17 MB |

    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        weights: one of `None` (random initialization), "imagenet" (pre-training on
            ImageNet), or the path to the weights file to be loaded.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.
    """
    return QuickNetFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


def QuickNetLarge(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the QuickNetLarge architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    quicknet_large-v1.0.0/quicknet_large.json
    ```
    ```summary
    sota.QuickNetLarge
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 66.0 %         | 87.0 %         | 23 342 248 | 5.40 MB |

    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        weights: one of `None` (random initialization), "imagenet" (pre-training on
            ImageNet), or the path to the weights file to be loaded.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.
    """
    return QuickNetLargeFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


def QuickNetSmall(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the QuickNetSmall architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    quicknet_small-v1.0.0/quicknet_small.json
    ```
    ```summary
    sota.QuickNetSmall
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 59.4 %         | 81.8 %         | 12 655 688 | 4.00 MB |

    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        weights: one of `None` (random initialization), "imagenet" (pre-training on
            ImageNet), or the path to the weights file to be loaded.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.
    """
    return QuickNetSmallFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
