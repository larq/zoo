from typing import Optional, Sequence

import larq as lq
import tensorflow as tf
from zookeeper import factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


@lq.utils.set_precision(1)
@lq.utils.register_keras_custom_object
def xnor_weight_scale(x):
    """
    Clips the weights between -1 and +1 and then calculates a scale factor per
    weight filter. See https://arxiv.org/abs/1603.05279 for more details
    """
    x = tf.clip_by_value(x, -1, 1)
    alpha = tf.reduce_mean(tf.abs(x), axis=[0, 1, 2], keepdims=True)
    return alpha * lq.quantizers.ste_sign(x)


@factory
class XNORNetFactory(ModelFactory):
    """Implementation of [XNOR-Net](https://arxiv.org/abs/1603.05279)"""

    input_quantizer = "ste_sign"
    kernel_quantizer = "xnor_weight_scale"
    kernel_constraint = "weight_clip"

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(5e-7)

    def build(self) -> tf.keras.models.Model:
        quant_conv_kwargs = dict(
            kernel_quantizer=self.kernel_quantizer,
            input_quantizer=self.input_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
        )

        x = tf.keras.layers.Conv2D(
            96,
            (11, 11),
            strides=(4, 4),
            padding="same",
            use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
        )(self.image_input)

        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(
            x
        )
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False, epsilon=1e-4)(
            x
        )
        x = lq.layers.QuantConv2D(256, (5, 5), padding="same", **quant_conv_kwargs)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False, epsilon=1e-4)(
            x
        )
        x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **quant_conv_kwargs)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False, epsilon=1e-4)(
            x
        )
        x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **quant_conv_kwargs)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False, epsilon=1e-4)(
            x
        )
        x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **quant_conv_kwargs)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False, epsilon=1e-4)(
            x
        )
        x = lq.layers.QuantConv2D(4096, (6, 6), padding="valid", **quant_conv_kwargs)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=False, epsilon=1e-4)(
            x
        )

        if self.include_top:
            # Equivalent to a dense layer
            x = lq.layers.QuantConv2D(
                4096, (1, 1), strides=(1, 1), padding="valid", **quant_conv_kwargs
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=0.9, scale=False, epsilon=1e-3
            )(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(
                self.num_classes,
                use_bias=False,
                kernel_regularizer=self.kernel_regularizer,
            )(x)
            x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        model = tf.keras.models.Model(
            inputs=self.image_input, outputs=x, name="xnornet"
        )

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="xnornet",
                    version="v0.2.0",
                    file="xnornet_weights.h5",
                    file_hash="e6ba24f785655260ae76a2ef1fab520e3528243d9c8fac430299cd81dbeabe10",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="xnornet",
                    version="v0.2.1",
                    file="xnornet_weights_notop.h5",
                    file_hash="20a17423090b2c80c6b7a6b62346faa4b2b7dc8d4da99efa792c9351cf86c3d5",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


def XNORNet(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
):
    """Instantiates the XNOR-Net architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    xnornet-v0.2.0/xnornet.json
    ```
    ```summary
    literature.XNORNet
    ```
    ```plot-altair
    /plots/xnornet.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 44.96 %        | 69.18 %        | 62 396 768 | 22.81 MB |

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
        - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural
            Networks](https://arxiv.org/abs/1603.05279)
    """
    return XNORNetFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
