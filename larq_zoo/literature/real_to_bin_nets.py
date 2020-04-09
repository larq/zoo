"""
This file implements the networks described in `Training binary neural networks with 
real-to-binary convolutions`
[(Martinez et al., 2019)](https://openreview.net/forum?id=BJg4NgBKvH)
"""

from typing import Optional, Any

import larq as lq
import tensorflow as tf
from zookeeper import component
from zookeeper.tf import Dataset, Model
from larq_zoo.model_factory import ModelFactory
from zookeeper import Field, factory


class _SharedBaseFactory(ModelFactory):
    """Base configuration and blocks shared across ResNet and Real-to-Bin Nets."""

    model_name: str = Field()

    momentum: float = Field(0.99)
    kernel_initializer: str = Field("glorot_normal")
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = Field(None)

    def _first_block(self, x, use_prelu=True, name=""):
        """First block, shared across ResNet and Real-to-Bin Nets."""

        x = tf.keras.layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            padding="same",
            name=f"{name}_conv2d",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=self.momentum, name=f"{name}_batch_norm"
        )(x)

        # Binary blocks use PReLU, ResNet blocks use ReLU
        if use_prelu:
            x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
        else:
            x = tf.keras.layers.Activation("relu", name=f"{name}_relu")(x)

        return tf.keras.layers.MaxPool2D(
            3, strides=2, padding="same", name=f"{name}_pool"
        )(x)

    def _last_block(self, x, name=""):
        """Last block, shared across ResNet and Real-to-Bin Nets."""

        x = tf.keras.layers.GlobalAvgPool2D(name=f"{name}_global_pool")(x)
        x = tf.keras.layers.Dense(
            self.num_classes, activation=None, name=f"{name}_logits"
        )(x)
        return tf.keras.layers.Activation(activation="softmax", name=f"{name}_probs")(x)

    def _block(self, x, downsample=False, name=""):
        """Main network block, different between ResNet and Real-to-Bin Nets.

        This is left to be implemented by the _ResNet18 and _RealToBinNet subclasses.
        """

        raise NotImplementedError()

    def _shortcut_connection(
        self, x: tf.Tensor, name: str, in_channels: int, out_channels: int
    ):
        if in_channels == out_channels:
            return x
        x = tf.keras.layers.AvgPool2D(
            2, strides=2, padding="same", name=f"{name}_shortcut_pool"
        )(x)
        x = tf.keras.layers.Conv2D(
            out_channels,
            1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False,
            name=f"{name}_shortcut_conv2d",
        )(x)
        return tf.keras.layers.BatchNormalization(
            momentum=self.momentum, name=f"{name}_shortcut_batch_norm"
        )(x)

    def build(self) -> tf.keras.models.Model:
        """Build a ResNet or Real-to-Bin Net."""

        x = self._first_block(
            self.image_input,
            name=f"{self.model_name}_block_1",
            use_prelu=isinstance(self, _StrongBaselineNet),
        )
        for block in range(2, 10):
            x = self._block(
                x,
                name=f"{self.model_name}_block_{block}",
                downsample=block % 2 == 0 and block > 3,
            )

        if self.include_top:
            x = self._last_block(x, name=f"{self.model_name}_block_10")

        model = tf.keras.Model(
            inputs=self.image_input, outputs=x, name=self.model_name,
        )

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                # TODO: FIXME
                raise NotImplementedError
                # weights_path = utils.download_pretrained_model(
                #     model="resnet_e",
                #     version="v0.1.0",
                #     file="resnet_e_18_weights.h5",
                #     file_hash="bde4a64d42c164a7b10a28debbe1ad5b287c499bc0247ecb00449e6e89f3bf5b",
                # )
            else:
                # TODO: FIXME
                raise NotImplementedError
                # weights_path = utils.download_pretrained_model(
                #     model="resnet_e",
                #     version="v0.1.0",
                #     file="resnet_e_18_weights_notop.h5",
                #     file_hash="14cb037e47d223827a8d09db88ec73d60e4153a4464dca847e5ae1a155e7f525",
                # )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


@factory
class StrongBaselineNetFactory(_SharedBaseFactory):
    """Constructor for the strong baseline (Section 4.1 of Martinez et al.)."""

    scaling_r: int = 8

    input_quantizer: Optional[str] = None
    kernel_quantizer: Optional[str] = None
    binary_kernel_constraint: Optional[Any] = None

    class LearnedRescaleLayer(tf.keras.layers.Layer):
        """Implements the learned activation rescaling XNOR-Net++ style.

        This is used to scale the outputs of the binary convolutions in the Strong
        Baseline networks.
        [(Bulat & Tzimiropoulos, 2019)](https://arxiv.org/abs/1909.13863)
        """

        def __init__(self, output_dim: tuple, **kwargs):
            super().__init__(**kwargs)
            assert (
                len(output_dim) == 3
            ), f"expected 3 dimensional output size, got {len(output_dim)} dimensional instead"
            self.output_dim = output_dim

        def build(self, input_shapes):
            self.scale_h = self.add_weight(
                name="scale_h",
                shape=(self.output_dim[0], 1, 1),
                initializer="ones",
                trainable=True,
            )
            self.scale_w = self.add_weight(
                name="scale_w",
                shape=(1, self.output_dim[1], 1),
                initializer="ones",
                trainable=True,
            )
            self.scale_c = self.add_weight(
                name="scale_c",
                shape=(1, 1, self.output_dim[2]),
                initializer="ones",
                trainable=True,
            )

            super().build(input_shapes)

        def call(self, inputs, **kwargs):
            return inputs * (self.scale_h * self.scale_w * self.scale_c)

        def compute_output_shape(self, **kwargs):
            return self.output_dim

        def get_config(self):
            return {"output_dim": self.output_dim}

    def _scale_binary_conv_output(
        self, conv_input: tf.Tensor, conv_output: tf.Tensor, name: str
    ):
        return LearnedRescaleLayer(conv_output.shape[1:], name=f"{name}_rescale")(
            conv_output
        )

    def _half_binary_block(
        self, x: tf.Tensor, downsample: bool = False, name: str = ""
    ):
        """One half of the binary block from Figure 1 (Left) of Martinez et al. (2019).

        This block gets repeated and matched up with/supervised by a single real block,
        which has two convolutions.

        Channel scaling follows Figure 1 (Right).
        """

        in_channels = x.shape[-1]
        out_channels = in_channels * 2 if downsample else in_channels

        # Shortcut, which gets downsampled if necessary
        shortcut_add = self._shortcut_connection(x, name, in_channels, out_channels)

        # Batch Normalization
        conv_input = tf.keras.layers.BatchNormalization(
            momentum=self.momentum, name=f"{name}_batch_norm"
        )(x)

        # Convolution
        conv_output = lq.layers.QuantConv2D(
            out_channels,
            3,
            strides=2 if downsample else 1,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
            name=f"{name}_conv2d",
        )(conv_input)

        # binary convolution rescaling
        x = self._scale_binary_conv_output(conv_input, conv_output, name)

        # PReLU activation
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)

        # Skip connection
        return tf.keras.layers.Add(name=f"{name}_skip_add")([x, shortcut_add])

    def _block(self, x, downsample=False, name=""):
        """Full binary block from Figure 1 (Left) of Matrinez et al. (2019)."""

        x = self._half_binary_block(x, downsample=downsample, name=f"{name}a")
        x = self._half_binary_block(x, downsample=False, name=f"{name}b")

        # Add explicit name to the block output for attention matching (Section 4.2 of Martinez et al.)
        return tf.keras.layers.Lambda(lambda x: x, name=f"{name}_out")(x)


@factory
class RealToBinNetFactory(_SharedBaseFactory):
    def _scale_binary_conv_output(
        self, conv_input: tf.Tensor, conv_output: tf.Tensor, name: str
    ):
        in_filters = conv_input.shape[-1]
        out_filters = conv_output.shape[-1]

        z = tf.keras.layers.GlobalAvgPool2D(name=f"{name}_scaling_pool")(conv_input)
        dim_reduction = tf.keras.layers.Dense(
            int(in_filters / self.scaling_r),
            activation="relu",
            kernel_initializer="he_normal",
            name=f"{name}_scaling_dense_reduce",
            use_bias=False,
        )(z)
        dim_expansion = tf.keras.layers.Dense(
            out_filters,
            activation="sigmoid",
            kernel_initializer="he_normal",
            name=f"{name}_scaling_dense_expand",
            use_bias=False,
        )(dim_reduction)
        scales = tf.keras.layers.Reshape(
            (1, 1, out_filters), name=f"{name}_scaling_reshape"
        )(dim_expansion)

        return tf.keras.layers.Multiply(name=f"{name}_scaling_multiplication")(
            [conv_output, scales]
        )


@factory
class ResNet18Factory(_SharedBaseFactory):
    """Constructor for a ResNet18 with layer names matching Real-to-Bin nets."""

    def _block(self, x, downsample=False, name=""):
        """One full residual block, consisting of two convolutions.

        This follows the definition of a "block" from Figure 1 (Left) of Martinez et al.
        (2019).
        """

        in_channels = x.shape[-1]
        out_channels = in_channels * 2 if downsample else in_channels

        # Shortcut, which gets downsampled if necessary
        shortcut_add = self._shortcut_connection(x, name, in_channels, out_channels)

        for convolution in ["a", "b"]:
            x = tf.keras.layers.Conv2D(
                out_channels,
                3,
                strides=2 if downsample and convolution == "a" else 1,
                padding="same",
                name=f"{name}{convolution}_conv2d",
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=self.momentum, name=f"{name}{convolution}_batch_norm"
            )(x)
            x = tf.keras.layers.Activation("relu", name=f"{name}{convolution}_relu")(x)

        x = tf.keras.layers.Add(name=f"{name}_skip_add")([x, shortcut_add])
        return tf.keras.layers.Activation("relu", name=f"{name}_out")(x)


def StrongBaselineNetBAN(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """TODO: Comment"""

    return StrongBaselineNetFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
        model_name="baseline_ban",
        input_quantizer="ste_sign",
    ).build()


"""Below are still TODO."""


@component
class StrongBaselineNetBNN(StrongBaselineNet):
    model_name = "baseline_bnn"
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"


@component
class RealToBinNetFP(RealToBinNet):
    model_name = "r2b_fp"
    input_quantizer = tf.keras.layers.Activation("tanh")  # soft binarization


@component
class RealToBinNetBAN(RealToBinNet):
    model_name = "r2b_ban"
    input_quantizer = "ste_sign"


@component
class RealToBinNetBNN(RealToBinNet):
    model_name = "r2b_bnn"
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"


@component
class ResNet18FP(ResNet18):
    model_name = "resnet_fp"
    input_quantizer = None
