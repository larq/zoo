from typing import Optional, Sequence

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


@factory
class BinaryResNetE18Factory(ModelFactory):
    """Implementation of [BinaryResNetE18](https://arxiv.org/abs/1906.08637)"""

    num_layers: int = Field(18)
    initial_filters: int = Field(64)

    @property
    def input_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_constraint(self):
        return lq.constraints.WeightClip(clip_value=1.25)

    @property
    def spec(self):
        spec = {
            18: ([2, 2, 2, 2], [64, 128, 256, 512]),
            34: ([3, 4, 6, 3], [64, 128, 256, 512]),
            50: ([3, 4, 6, 3], [256, 512, 1024, 2048]),
            101: ([3, 4, 23, 3], [256, 512, 1024, 2048]),
            152: ([3, 8, 36, 3], [256, 512, 1024, 2048]),
        }
        try:
            return spec[self.num_layers]
        except Exception:
            raise ValueError(f"Only specs for layers {list(self.spec.keys())} defined.")

    def residual_block(self, x: tf.Tensor, filters: int, strides: int = 1) -> tf.Tensor:
        downsample = x.get_shape().as_list()[-1] != filters

        if downsample:
            residual = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)(x)
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

        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_initializer="glorot_normal",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        return tf.keras.layers.add([x, residual])

    def build(self) -> tf.keras.models.Model:
        if self.image_input.shape[1] and self.image_input.shape[1] < 50:
            x = tf.keras.layers.Conv2D(
                self.initial_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            )(self.image_input)
        else:
            x = tf.keras.layers.Conv2D(
                self.initial_filters,
                kernel_size=7,
                strides=2,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            )(self.image_input)

            x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        for block, (layers, filters) in enumerate(zip(*self.spec)):
            # This trick adds shortcut connections between original ResNet
            # blocks. We wultiply the number of blocks by two, but add only one
            # layer instead of two in each block
            for layer in range(layers * 2):
                strides = 1 if block == 0 or layer != 0 else 2
                x = self.residual_block(x, filters, strides=strides)

        x = tf.keras.layers.Activation("relu")(x)

        if self.include_top:
            x = utils.global_pool(x)
            x = tf.keras.layers.Dense(
                self.num_classes, kernel_initializer="glorot_normal"
            )(x)
            x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        model = tf.keras.Model(
            inputs=self.image_input,
            outputs=x,
            name=f"binary_resnet_e_{self.num_layers}",
        )

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="resnet_e",
                    version="v0.1.0",
                    file="resnet_e_18_weights.h5",
                    file_hash="bde4a64d42c164a7b10a28debbe1ad5b287c499bc0247ecb00449e6e89f3bf5b",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="resnet_e",
                    version="v0.1.0",
                    file="resnet_e_18_weights_notop.h5",
                    file_hash="14cb037e47d223827a8d09db88ec73d60e4153a4464dca847e5ae1a155e7f525",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


def BinaryResNetE18(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the BinaryResNetE 18 architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    resnet_e-v0.1.0/resnet_e_18.json
    ```
    ```summary
    literature.BinaryResNetE18
    ```
    ```plot-altair
    /plots/resnet_e_18.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 58.32 %        | 80.79 %        | 11 699 368 | 4.03 MB |

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

    # References
        - [Back to Simplicity: How to Train Accurate BNNs from
            Scratch?](https://arxiv.org/abs/1906.08637)
    """
    return BinaryResNetE18Factory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
