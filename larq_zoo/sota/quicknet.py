import abc
from typing import Callable, Optional, Sequence

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


def stem_module(filters: int, x: tf.Tensor) -> tf.Tensor:
    assert filters % 8 == 0
    x = tf.keras.layers.Conv2D(
        filters // 8,
        (3, 3),
        strides=2,
        kernel_initializer="he_normal",
        padding="same",
        use_bias=False,
    )(x)
    x = tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        depth_multiplier=8,
        strides=2,
        kernel_initializer="he_normal",
        padding="same",
        use_bias=False,
        activation="relu",
    )(x)

    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)


def squeeze_and_excite(inp: tf.Tensor, filters: int, r: int = 16):
    """Squeeze and Excite as per [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).

    Use of S&E in BNNs was pioneered in [Training binary neural networks with
    real-to-binary convolutions](https://openreview.net/forum?id=BJg4NgBKvH).
    """
    out = utils.global_pool(inp)
    out = tf.keras.layers.Dense(
        inp.shape[-1] // r,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(out)
    out = tf.keras.layers.Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(out)

    return tf.reshape(out, [-1, 1, 1, filters])


class QuickNetBaseFactory(ModelFactory, abc.ABC):
    name: str = "model"
    blocks_per_section: Sequence[int] = Field()
    section_filters: Sequence[int] = Field()
    use_squeeze_and_excite_in_section: Sequence[bool] = Field()
    transition_block: Callable[..., tf.Tensor] = Field()
    stem_filters: int = Field(64)

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
        assert (
            len(self.blocks_per_section)
            == len(self.section_filters)
            == len(self.use_squeeze_and_excite_in_section)
        )

    @property
    @abc.abstractmethod
    def imagenet_weights_path(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def imagenet_no_top_weights_path(self) -> str:
        raise NotImplementedError

    def conv_block(
        self, x: tf.Tensor, filters: int, use_squeeze_and_excite: bool, strides: int = 1
    ) -> tf.Tensor:
        """Convolution, batch norm and optionally squeeze & excite attention module."""
        if use_squeeze_and_excite:
            y = squeeze_and_excite(x, filters)
        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=3,
            strides=strides,
            padding="Same",
            pad_values=1.0,
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_initializer="glorot_normal",
            use_bias=False,
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        if use_squeeze_and_excite:
            x *= y

        return x

    def residual_block(self, x: tf.Tensor, use_squeeze_and_excite: bool) -> tf.Tensor:
        """Standard residual block, without strides or filter changes."""
        infilters = int(x.shape[-1])
        residual = x
        x = self.conv_block(x, infilters, use_squeeze_and_excite)
        return tf.keras.layers.add([x, residual])

    def concat_transition_block(
        self, x: tf.Tensor, filters: int, strides: int, use_squeeze_and_excite: bool
    ) -> tf.Tensor:
        """Concat transition block.

        Doubles the number of filters by concatenating shortcut with x + shortcut.
        """
        infilters = int(x.shape[-1])
        assert filters == 2 * infilters

        residual = tf.keras.layers.MaxPool2D(pool_size=strides, strides=strides)(x)
        residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
            residual
        )
        x = self.conv_block(x, infilters, use_squeeze_and_excite, strides)
        x = tf.keras.layers.add([x, residual])

        return tf.keras.layers.concatenate([residual, x])

    def fp_pointwise_transition_block(
        self, x: tf.Tensor, filters: int, strides: int, use_squeeze_and_excite: bool
    ) -> tf.Tensor:
        """Pointwise transition block.

        Transition to arbitrary number of filters by inserting pointwise
        full-precision convolution in shortcut.
        """
        residual = tf.keras.layers.MaxPool2D(pool_size=strides, strides=strides)(x)
        residual = tf.keras.layers.Conv2D(
            filters, kernel_size=1, use_bias=False, kernel_initializer="glorot_normal"
        )(residual)
        residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
            residual
        )
        x = self.conv_block(x, filters, use_squeeze_and_excite, strides)
        return tf.keras.layers.add([x, residual])

    def build(self) -> tf.keras.models.Model:
        x = stem_module(self.stem_filters, self.image_input)

        for block, (layers, filters, use_squeeze_and_excite) in enumerate(
            zip(
                self.blocks_per_section,
                self.section_filters,
                self.use_squeeze_and_excite_in_section,
            )
        ):
            for layer in range(layers):
                if filters == x.shape[-1]:
                    x = self.residual_block(x, use_squeeze_and_excite)
                else:
                    strides = 1 if (block == 0 or layer != 0) else 2
                    x = self.transition_block(
                        x, filters, strides, use_squeeze_and_excite
                    )

        x = tf.keras.layers.Activation("relu")(x)

        if self.include_top:
            x = utils.global_pool(x)
            x = tf.keras.layers.Dense(
                self.num_classes, kernel_initializer="glorot_normal"
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
class QuickNetFactory(QuickNetBaseFactory):
    """Quicknet - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)"""

    name = "quicknet"
    blocks_per_section: Sequence[int] = Field((2, 3, 4, 4))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))
    use_squeeze_and_excite_in_section: Sequence[bool] = Field(
        (False, False, False, False)
    )
    transition_block = Field(lambda self: self.concat_transition_block)

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v0.2.1",
            file="quicknet_weights.h5",
            file_hash="7b4fa94f5241c7aad3412ca42b5db6517dbc4847cff710cb82be10c2f83bc0be",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet",
            version="v0.2.1",
            file="quicknet_weights_notop.h5",
            file_hash="359eed6dae43525eddf520ea87ec9b54750ee0e022647775d115a38856be396f",
        )


@factory
class QuickNetLargeFactory(QuickNetBaseFactory):
    """QuickNetLarge - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)
    and high accuracy. This utilises Squeeze and Excite blocks as per [Training binary neural networks with real-to-binary convolutions](https://openreview.net/forum?id=BJg4NgBKvH)."""

    name = "quicknet_large"
    blocks_per_section: Sequence[int] = Field((4, 4, 4, 4))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))
    use_squeeze_and_excite_in_section: Sequence[bool] = Field(
        (False, False, True, True)
    )
    transition_block = Field(lambda self: self.fp_pointwise_transition_block)

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_large",
            version="v0.2.1",
            file="quicknet_large_weights.h5",
            file_hash="6bf778e243466c678d6da0e3a91c77deec4832460046fca9e6ac8ae97a41299c",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_large",
            version="v0.2.1",
            file="quicknet_large_weights_notop.h5",
            file_hash="b65d59dd2d5af63d019997b05faff9e003510e2512aa973ee05eb1b82b8792a9",
        )


@factory
class QuickNetXLFactory(QuickNetBaseFactory):
    """QuickNetXL - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)
    and high accuracy. This utilises Squeeze and Excite blocks as per [Training binary neural networks with real-to-binary convolutions](https://openreview.net/forum?id=BJg4NgBKvH)."""

    name = "quicknet_xl"
    blocks_per_section: Sequence[int] = Field((6, 8, 12, 6))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))
    use_squeeze_and_excite_in_section: Sequence[bool] = Field(
        (False, False, True, True)
    )
    transition_block = Field(lambda self: self.fp_pointwise_transition_block)

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_xl",
            version="v0.1.1",
            file="quicknet_xl_weights.h5",
            file_hash="19a41e753dbd4fbc3cbdaecd3627fb536ef55d64702996aae3875a8de3cf8073",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="quicknet_xl",
            version="v0.1.1",
            file="quicknet_xl_weights_notop.h5",
            file_hash="ad5cbfa333b0aabde75dc524c9ce4a5ae096061da0e2dcf362ec6e587a83a511",
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
    quicknet-v0.2.0/quicknet.json
    ```
    ```summary
    sota.QuickNet
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 58.6 %         | 81.0 %         | 10 518 528 | 3.21 MB |

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
    quicknet_large-v0.2.0/quicknet_large.json
    ```
    ```summary
    sota.QuickNetLarge
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 62.7 %         | 84.0 %         | 11 837 696 | 4.56 MB |

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


def QuickNetXL(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the QuickNetXL architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    quicknet_xl-v0.1.0/quicknet_xl.json
    ```
    ```summary
    sota.QuickNetXL
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 67.0 %         | 87.3 %         | 22 058 368 | 6.22 MB |

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
    return QuickNetXLFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
