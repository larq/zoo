from typing import Optional, Sequence

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo import utils
from larq_zoo.model_factory import ModelFactory


@lq.utils.register_keras_custom_object
@lq.utils.set_precision(1)
def magnitude_aware_sign_unclipped(x):
    """
    Scaled sign function with identity pseudo-gradient as used for the weights
    in the DoReFa paper. The Scale factor is calculated per layer.
    """
    scale_factor = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

    @tf.custom_gradient
    def _magnitude_aware_sign(x):
        return lq.math.sign(x) * scale_factor, lambda dy: dy

    return _magnitude_aware_sign(x)


@lq.utils.register_keras_custom_object
def clip_by_value_activation(x):
    return tf.clip_by_value(x, 0, 1)


@factory
class DoReFaNetFactory(ModelFactory):
    """
    Implementation of [DoReFa-Net: Training Low Bitwidth Convolutional Neural
    Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
    """

    activations_k_bit: int = Field(2)

    input_quantizer = Field(
        lambda self: lq.quantizers.DoReFaQuantizer(k_bit=self.activations_k_bit)
    )
    kernel_quantizer = Field(lambda: magnitude_aware_sign_unclipped)
    kernel_constraint = Field(None)

    def conv_block(
        self, x, filters, kernel_size, strides=1, pool=False, pool_padding="same"
    ):
        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9, epsilon=1e-4)(
            x
        )
        if pool:
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding=pool_padding)(
                x
            )
        return x

    def fully_connected_block(self, x, units):
        x = lq.layers.QuantDense(
            units,
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        return tf.keras.layers.BatchNormalization(
            scale=False, momentum=0.9, epsilon=1e-4
        )(x)

    def build(self) -> tf.keras.models.Model:
        out = tf.keras.layers.Conv2D(
            96, kernel_size=12, strides=4, padding="valid", use_bias=True
        )(self.image_input)
        out = self.conv_block(out, filters=256, kernel_size=5, pool=True)
        out = self.conv_block(out, filters=384, kernel_size=3, pool=True)
        out = self.conv_block(out, filters=384, kernel_size=3)
        out = self.conv_block(
            out, filters=256, kernel_size=3, pool_padding="valid", pool=True
        )

        if self.include_top:
            out = tf.keras.layers.Flatten()(out)
            out = self.fully_connected_block(out, units=4096)
            out = self.fully_connected_block(out, units=4096)
            out = tf.keras.layers.Activation("clip_by_value_activation")(out)
            out = tf.keras.layers.Dense(self.num_classes, use_bias=True)(out)
            out = tf.keras.layers.Activation("softmax")(out)

        model = tf.keras.Model(inputs=self.image_input, outputs=out, name="dorefanet")

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="dorefanet",
                    version="v0.1.0",
                    file="dorefanet_weights.h5",
                    file_hash="645d7839d574faa3eeeca28f3115773d75da3ab67ff6876b4de12d10245ecf6a",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="dorefanet",
                    version="v0.1.0",
                    file="dorefanet_weights_notop.h5",
                    file_hash="679368128e19a2a181bfe06ca3a3dec368b1fd8011d5f42647fbbf5a7f36d45f",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


def DoReFaNet(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the DoReFa-net architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    dorefanet-v0.1.0/dorefanet.json
    ```
    ```plot-altair
    /plots/dorefanet.vg.json
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
    num_classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is specified.

    # Returns
    A Keras model instance.

    # Raises
    ValueError: in case of invalid argument for `weights`, or invalid input shape.

    # References
    - [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low
    Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
    """
    return DoReFaNetFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
