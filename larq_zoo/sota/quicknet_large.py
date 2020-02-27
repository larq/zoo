from typing import Optional, Sequence, Tuple

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo import utils
from larq_zoo.model_factory import ModelFactory
from larq_zoo.sota.quicknet import LCEFirstLayer


def squeeze_and_excite(inp: tf.Tensor, strides: int = 1, r: int = 16):
    """Squeeze and Excite as per [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)"""
    C = inp.get_shape().as_list()[-1]

    out = tf.keras.layers.GlobalAvgPool2D()(inp)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(
        C // r,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(out)
    outmult = 2 if strides == 2 else 1
    out = tf.keras.layers.Dense(
        C * outmult,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(out)
    return tf.reshape(out, [-1, 1, 1, C * outmult])


@factory
class QuickNetLargeFactory(ModelFactory):
    """QuickNetLarge - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)
    and high accuracy. This utilises Squeeze and Excite blocks as per [Training binary neural networks with real-to-binary convolutions](https://openreview.net/forum?id=BJg4NgBKvH)."""

    num_layers: int = Field(18)
    initial_filters: int = Field(64)

    input_quantizer = Field(lambda: lq.quantizers.SteSign(clip_value=1.25))
    kernel_quantizer = Field(lambda: lq.quantizers.SteSign(clip_value=1.25))
    kernel_constraint = Field(lambda: lq.constraints.WeightClip(clip_value=1.25))

    @Field
    def spec(self) -> Tuple[Sequence[int], Sequence[int]]:
        spec = {18: ([4, 4, 4, 4], [64, 128, 256, 512])}
        try:
            return spec[self.num_layers]
        except Exception:
            raise ValueError(f"Only specs for layers {list(self.spec.keys())} defined.")

    def residual_block_SE(
        self, x: tf.Tensor, filters: int, strides: int = 1
    ) -> tf.Tensor:
        downsample = x.get_shape().as_list()[-1] != filters

        if downsample and strides == 2:
            residual = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
            residual = tf.keras.layers.Conv2D(
                filters,
                kernel_size=1,
                use_bias=False,
                kernel_initializer="glorot_normal",
            )(residual)
            residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
                residual
            )
        else:
            residual = x

        y = squeeze_and_excite(x, strides=strides)
        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=3,
            strides=strides,
            padding="Same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_initializer="glorot_normal",
            use_bias=False,
            activation="relu",
            metrics=[],
        )(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = tf.multiply(x, y)

        return tf.keras.layers.add([x, residual])

    def build(self) -> tf.keras.models.Model:
        x = LCEFirstLayer(self.initial_filters, self.image_input)

        for block, (layers, filters) in enumerate(zip(*self.spec)):
            for layer in range(layers):
                strides = 1 if block == 0 or layer != 0 else 2
                x = self.residual_block_SE(x, filters, strides=strides)

        x = tf.keras.layers.Activation("relu")(x)
        if self.include_top:
            x = tf.keras.layers.GlobalAvgPool2D()(x)
            x = tf.keras.layers.Dense(
                self.num_classes,
                activation="softmax",
                kernel_initializer="glorot_normal",
            )(x)

        model = tf.keras.Model(inputs=self.image_input, outputs=x, name="quicknet")

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="quicknet_large",
                    version="v0.1.0",
                    file="quicknet_large_weights.h5",
                    file_hash="1cd7ff411710023f902ec0152b860b4c3dea82e4bfe373b9310ca3598b3de640",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="quicknet_large",
                    version="v0.1.0",
                    file="quicknet_large_weights_notop.h5",
                    file_hash="628bde2a21cf9338e90da9b0179432d1dfa5f25a7d2aa6e51fc91d1630675c10",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


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
    quicknet_large-v0.1.0/quicknet_large.json
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
