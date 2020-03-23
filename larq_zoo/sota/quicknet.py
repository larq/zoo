from types import MethodType
from typing import Optional, Sequence

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


@factory
class QuickNetBaseFactory(ModelFactory):

    blocks_per_section: Sequence[int] = Field(None)
    section_filters: Sequence[int] = Field(None)
    use_squeeze_and_excite_in_section: Sequence[bool] = Field(None)
    transition_block: MethodType = Field(None)
    stem_filters: int = Field(64)

    input_quantizer = Field(lambda: lq.quantizers.SteSign(clip_value=1.25))
    kernel_quantizer = Field(lambda: lq.quantizers.SteSign(clip_value=1.25))
    kernel_constraint = Field(lambda: lq.constraints.WeightClip(clip_value=1.25))

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

        Doubles number of filters by concatenating shortcut with x + shortcut.
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

        model = tf.keras.Model(inputs=self.image_input, outputs=x, name="quicknet")

        return model


@factory
class QuickNetFactory(QuickNetBaseFactory):
    """Quicknet - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)"""

    blocks_per_section: Sequence[int] = Field((2, 3, 4, 4))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))
    use_squeeze_and_excite_in_section: Sequence[bool] = Field(
        (False, False, False, False)
    )
    transition_block = Field(lambda self: self.concat_transition_block)

    def build(self) -> tf.keras.models.Model:
        model = super().build()

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="quicknet",
                    version="v0.2.0",
                    file="quicknet_weights.h5",
                    file_hash="6a765f120ba7b62a7740e842c4f462eb7ba3dd65eb46b4694c5bc8169618fae7",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="quicknet",
                    version="v0.2.0",
                    file="quicknet_weights_notop.h5",
                    file_hash="5bf2fc450fb8cc322b33a16410bf88fed09d05c221550c2d5805a04985383ac2",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


@factory
class QuickNetLargeFactory(QuickNetBaseFactory):
    """QuickNetLarge - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)
    and high accuracy. This utilises Squeeze and Excite blocks as per [Training binary neural networks with real-to-binary convolutions](https://openreview.net/forum?id=BJg4NgBKvH)."""

    blocks_per_section: Sequence[int] = Field((4, 4, 4, 4))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))
    use_squeeze_and_excite_in_section: Sequence[bool] = Field(
        (False, False, True, True)
    )
    transition_block = Field(lambda self: self.fp_pointwise_transition_block)

    def build(self) -> tf.keras.models.Model:
        model = super().build()
        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="quicknet_large",
                    version="v0.2.0",
                    file="quicknet_large_weights.h5",
                    file_hash="2d9ebbf8ba0500552e4dd243c3e52fd8291f965ef6a0e1dbba13cc72bf6eee8b",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="quicknet_large",
                    version="v0.2.0",
                    file="quicknet_large_weights_notop.h5",
                    file_hash="067655ef8a1a1e99ef1c71fa775c09aca44bdfad0b9b71538b4ec500c3beee4f",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


@factory
class QuickNetXLFactory(QuickNetBaseFactory):
    """QuickNetXL - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)
    and high accuracy. This utilises Squeeze and Excite blocks as per [Training binary neural networks with real-to-binary convolutions](https://openreview.net/forum?id=BJg4NgBKvH)."""

    blocks_per_section: Sequence[int] = Field((6, 8, 12, 6))
    section_filters: Sequence[int] = Field((64, 128, 256, 512))
    use_squeeze_and_excite_in_section: Sequence[bool] = Field(
        (False, False, True, True)
    )
    transition_block = Field(lambda self: self.fp_pointwise_transition_block)

    def build(self) -> tf.keras.models.Model:
        model = super().build()
        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="quicknet_xl",
                    version="v0.1.0",
                    file="quicknet_xl_weights.h5",
                    file_hash="a85eea1204fa9a8401f922f94531858493e3518e3374347978ed7ba615410498",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="quicknet_xl",
                    version="v0.1.0",
                    file="quicknet_xl_weights_notop.h5",
                    file_hash="b97074d6618acde4201d1f8676d32272d27743ddfe27c6c97e4516511ebb5008",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


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

    # Arguments
    input_shape: Optional shape tuple, to be specified if you would like to use a model
        with an input image resolution that is not (224, 224, 3).
        It should have exactly 3 inputs channels.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
        image input for the model.
    weights: one of `None` (random initialization), "imagenet" (pre-training on
        ImageNet), or the path to the weights file to be loaded.
    include_top: whether to include the fully-connected layer at the top of the network.
    num_classes: optional number of classes to classify images into, only to be specified
        if `include_top` is True, and if no `weights` argument is specified.

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

    # Arguments
    input_shape: Optional shape tuple, to be specified if you would like to use a model
        with an input image resolution that is not (224, 224, 3).
        It should have exactly 3 inputs channels.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
        image input for the model.
    weights: one of `None` (random initialization), "imagenet" (pre-training on
        ImageNet), or the path to the weights file to be loaded.
    include_top: whether to include the fully-connected layer at the top of the network.
    num_classes: optional number of classes to classify images into, only to be specified
        if `include_top` is True, and if no `weights` argument is specified.

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

    # Arguments
    input_shape: Optional shape tuple, to be specified if you would like to use a model
        with an input image resolution that is not (224, 224, 3).
        It should have exactly 3 inputs channels.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
        image input for the model.
    weights: one of `None` (random initialization), "imagenet" (pre-training on
        ImageNet), or the path to the weights file to be loaded.
    include_top: whether to include the fully-connected layer at the top of the network.
    num_classes: optional number of classes to classify images into, only to be specified
        if `include_top` is True, and if no `weights` argument is specified.

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
