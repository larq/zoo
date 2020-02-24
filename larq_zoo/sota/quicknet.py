from typing import Optional, Sequence, Tuple

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo import utils
from larq_zoo.model_factory import ModelFactory


def LCEFirstLayer(filters: int, x: tf.Tensor) -> tf.Tensor:
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


@factory
class QuickNetFactory(ModelFactory):
    """Quicknet - A model designed for fast inference using [Larq Compute Engine](https://github.com/larq/compute-engine)"""

    num_layers: int = Field(15)
    initial_filters: int = Field(64)

    input_quantizer = Field(lambda: lq.quantizers.SteSign(clip_value=1.25))
    kernel_quantizer = Field(lambda: lq.quantizers.SteSign(clip_value=1.25))
    kernel_constraint = Field(lambda: lq.constraints.WeightClip(clip_value=1.25))

    @Field
    def spec(self) -> Tuple[Sequence[int], Sequence[int]]:
        spec = {15: ([2, 3, 4, 4], [64, 128, 256, 512])}
        try:
            return spec[self.num_layers]
        except Exception:
            raise ValueError(f"Only specs for layers {list(self.spec.keys())} defined.")

    def residual_fast_block(
        self, x: tf.Tensor, filters: int, strides: int = 1
    ) -> tf.Tensor:
        """LCE Optimised residual block"""
        infilters = x.get_shape().as_list()[-1]
        downsample = infilters != filters

        if downsample and strides == 2:
            residual = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
            residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
                residual
            )
        else:
            residual = x

        x = lq.layers.QuantConv2D(
            infilters,
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
        if downsample:
            return tf.keras.layers.concatenate(
                [residual, tf.keras.layers.add([x, residual])]
            )
        else:
            return tf.keras.layers.add([x, residual])

    def build(self) -> tf.keras.models.Model:
        x = LCEFirstLayer(self.initial_filters, self.image_input)

        for block, (layers, filters) in enumerate(zip(*self.spec)):
            for layer in range(layers):
                strides = 1 if block == 0 or layer != 0 else 2
                x = self.residual_fast_block(x, filters, strides=strides)

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
                    model="quicknet",
                    version="v0.1.0",
                    file="quicknet_weights.h5",
                    file_hash="f52abb0ce984015889f8a8842944eed1bfad06897d745c7b58eb663b3457cd3c",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="quicknet",
                    version="v0.1.0",
                    file="quicknet_weights_notop.h5",
                    file_hash="057391ea350ce0af33194db300d3d9d690c8fb5b11427bbaf37504af257e9dc5",
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
    quicknet-v0.1.0/quicknet.json
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
