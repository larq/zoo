"""This file implements the networks described in `Training binary neural networks with
real-to-binary convolutions`

[(Martinez et al., 2019)](https://openreview.net/forum?id=BJg4NgBKvH)
"""
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


class _SharedBaseFactory(ModelFactory, metaclass=ABCMeta):
    """Base configuration and blocks shared across ResNet, StrongBaselineNets and Real-
    to-Bin Nets."""

    model_name: str = Field()
    momentum: float = Field(0.99)
    kernel_initializer: str = Field("glorot_normal")
    kernel_regularizer = None

    def first_block(
        self, x: tf.Tensor, use_prelu: bool = True, name: str = ""
    ) -> tf.Tensor:
        """First block, shared across ResNet, StrongBaselineNet and Real-to-Bin Nets."""

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

        # StrongBaselineNet uses PReLU; ResNets and Real-to-Bin nets use ReLU.
        if use_prelu:
            x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
        else:
            x = tf.keras.layers.ReLU(name=f"{name}_relu")(x)

        return tf.keras.layers.MaxPool2D(
            3, strides=2, padding="same", name=f"{name}_pool"
        )(x)

    def last_block(self, x: tf.Tensor, name: str = "") -> tf.Tensor:
        """Last block, shared across ResNet, StrongBaselineNet and Real-to-Bin nets."""

        x = utils.global_pool(x, name=f"{name}_global_pool")
        x = tf.keras.layers.Dense(
            self.num_classes,
            name=f"{name}_logits",
        )(x)
        return tf.keras.layers.Softmax(name=f"{name}_probs", dtype=tf.float32)(x)

    @abstractmethod
    def block(self, x, downsample=False, name=""):
        """Main network block

        This block differs between ResNet and StrongBaseline / Real-to-Bin Nets.
        It is implemented by the ResNet18 and StrongBaselineNet subclasses.
        """

    def shortcut_connection(
        self, x: tf.Tensor, name: str, in_channels: int, out_channels: int
    ) -> tf.Tensor:
        if in_channels == out_channels:
            return x
        x = tf.keras.layers.AvgPool2D(
            2, strides=2, padding="same", name=f"{name}_shortcut_pool"
        )(x)
        x = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False,
            name=f"{name}_shortcut_conv2d",
        )(x)
        return tf.keras.layers.BatchNormalization(
            momentum=self.momentum, name=f"{name}_shortcut_batch_norm"
        )(x)

    def build(self) -> tf.keras.models.Model:
        """Build the model."""

        x = self.first_block(
            self.image_input,
            name=f"{self.model_name}_block_1",
            use_prelu=isinstance(self, StrongBaselineNetFactory),
        )
        for block in range(2, 10):
            x = self.block(
                x,
                name=f"{self.model_name}_block_{block}",
                downsample=block % 2 == 0 and block > 3,
            )

        if self.include_top:
            x = self.last_block(x, name=f"{self.model_name}_block_10")

        model = tf.keras.Model(
            inputs=self.image_input,
            outputs=x,
            name=self.model_name,
        )

        # Load weights.
        if self.weights == "imagenet":
            model.load_weights(self._get_imagenet_weights_path())
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model

    def _get_imagenet_weights_path(self):
        raise ValueError(f"No stored weights for {self.model_name}")


@factory
class StrongBaselineNetFactory(_SharedBaseFactory):
    """Constructor for the strong baseline network (Section 4.1 of Martinez et al.)."""

    scaling_r: int = 8

    input_quantizer = None
    kernel_quantizer = None

    class LearnedRescaleLayer(tf.keras.layers.Layer):
        """Implements the learned activation rescaling XNOR-Net++ style.

        This is used to scale the outputs of the binary convolutions in the Strong
        Baseline networks. [(Bulat & Tzimiropoulos,
        2019)](https://arxiv.org/abs/1909.13863)
        """

        def __init__(
            self,
            regularizer: Optional[tf.keras.regularizers.Regularizer],
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.kernel_regularizer = tf.keras.regularizers.get(regularizer)

        def build(self, input_shapes):
            self.scale_h = self.add_weight(
                name="scale_h",
                shape=(input_shapes[1], 1, 1),
                initializer="ones",
                regularizer=self.kernel_regularizer,
                trainable=True,
            )
            self.scale_w = self.add_weight(
                name="scale_w",
                shape=(1, input_shapes[2], 1),
                initializer="ones",
                regularizer=self.kernel_regularizer,
                trainable=True,
            )
            self.scale_c = self.add_weight(
                name="scale_c",
                shape=(1, 1, input_shapes[3]),
                initializer="ones",
                regularizer=self.kernel_regularizer,
                trainable=True,
            )

            super().build(input_shapes)

        def call(self, inputs, **kwargs):
            return inputs * (self.scale_h * self.scale_w * self.scale_c)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            return {
                **super().get_config(),
                "regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            }

    def _scale_binary_conv_output(
        self, conv_input: tf.Tensor, conv_output: tf.Tensor, name: str
    ) -> tf.Tensor:
        """Flexible wrapper for the `LearnedRescaleLayer`.

        The way in which the output of the binary convolution is scaled is the only
        structural difference between the StrongBaseline networks and the Real-to-Binary
        networks. This function accepts all inputs required for either function.

        The Strong Baseline network uses the learned static rescale layer of
        Bulat & Tzimiropoulos
        """
        return self.LearnedRescaleLayer(
            regularizer=self.kernel_regularizer,
            name=f"{name}_rescale",
        )(conv_output)

    def half_binary_block(
        self, x: tf.Tensor, downsample: bool = False, name: str = ""
    ) -> tf.Tensor:
        """One half of the binary block from Figure 1 (Left) of Martinez et al. (2019).

        This block gets repeated and matched up with/supervised by a single real block,
        which has two convolutions.

        Channel scaling follows Figure 1 (Right).
        """

        in_channels = x.shape[-1]
        out_channels = int(in_channels * 2 if downsample else in_channels)

        # Shortcut, which gets downsampled if necessary
        shortcut_add = self.shortcut_connection(x, name, in_channels, out_channels)

        # Batch Normalization
        conv_input = tf.keras.layers.BatchNormalization(
            momentum=self.momentum, name=f"{name}_batch_norm"
        )(x)

        # Convolution
        conv_output = lq.layers.QuantConv2D(
            out_channels,
            kernel_size=3,
            strides=2 if downsample else 1,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_regularizer=self.kernel_regularizer
            if self.kernel_quantizer is None
            else None,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=f"{name}_conv2d",
        )(conv_input)

        # binary convolution rescaling
        x = self._scale_binary_conv_output(conv_input, conv_output, name)

        # PReLU activation
        x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)

        # Skip connection
        return tf.keras.layers.Add(name=f"{name}_skip_add")([x, shortcut_add])

    def block(
        self, x: tf.Tensor, downsample: bool = False, name: str = ""
    ) -> tf.Tensor:
        """Full binary block from Figure 1 (Left) of Matrinez et al. (2019)."""

        x = self.half_binary_block(x, downsample=downsample, name=f"{name}a")
        x = self.half_binary_block(x, downsample=False, name=f"{name}b")

        # Add explicit name to the block output for attention matching (Section 4.2 of
        # Martinez et al.)
        return tf.keras.layers.Lambda(lambda x: x, name=f"{name}_out")(x)


@factory
class RealToBinNetFactory(StrongBaselineNetFactory):
    def _scale_binary_conv_output(
        self, conv_input: tf.Tensor, conv_output: tf.Tensor, name: str
    ) -> tf.Tensor:
        """Data-dependent convolution scaling.

        Scales the output of the convolution in the (squeeze-and-excite
        style) data-dependent way described in Section 4.3 of Martinez at. al.
        """
        in_filters = conv_input.shape[-1]
        out_filters = conv_output.shape[-1]

        z = utils.global_pool(conv_input, name=f"{name}_scaling_pool")
        dim_reduction = tf.keras.layers.Dense(
            int(in_filters // self.scaling_r),
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            name=f"{name}_scaling_dense_reduce",
            use_bias=False,
        )(z)
        dim_expansion = tf.keras.layers.Dense(
            out_filters,
            activation="sigmoid",
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            name=f"{name}_scaling_dense_expand",
            use_bias=False,
        )(dim_reduction)
        scales = tf.keras.layers.Reshape(
            (1, 1, out_filters), name=f"{name}_scaling_reshape"
        )(dim_expansion)

        return tf.keras.layers.Multiply(name=f"{name}_scaling_multiplication")(
            [conv_output, scales]
        )

    def _get_imagenet_weights_path(self):
        if (
            not self.kernel_quantizer == "ste_sign"
            and self.input_quantizer == "ste_sign"
        ):
            raise ValueError(
                f"{self.model_name} only has ImageNet weights for the BNN variant"
            )
        if self.include_top:
            weights_path = utils.download_pretrained_model(
                model="r2b",
                version="v0.1.0",
                file="r2b_weights.h5",
                file_hash="e8fd16ca1ab9810ac3835f24f5c62758a57bc32a615f73aaa50d382d2b9617e1",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="r2b",
                version="v0.1.0",
                file="r2b_weights_notop.h5",
                file_hash="4ec47abf1a4da5c65f4908076257e8d5c812673891089a88c9d9e84e949d1dab",
            )
        return weights_path


@factory
class ResNet18Factory(_SharedBaseFactory):
    """Constructor for a ResNet18 with layer names matching Real-to-Bin nets."""

    def block(
        self, x: tf.Tensor, downsample: bool = False, name: str = ""
    ) -> tf.Tensor:
        """One full residual block, consisting of two convolutions.

        This follows the definition of a "block" from Figure 1 (Left) of Martinez et al.
        """

        in_channels = x.shape[-1]
        out_channels = int(in_channels * 2 if downsample else in_channels)

        # Shortcut, which gets downsampled if necessary
        shortcut_add = self.shortcut_connection(x, name, in_channels, out_channels)

        for convolution in ["a", "b"]:
            x = tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=3,
                kernel_regularizer=self.kernel_regularizer,
                strides=2 if downsample and convolution == "a" else 1,
                padding="same",
                name=f"{name}{convolution}_conv2d",
            )(x)
            x = tf.keras.layers.BatchNormalization(
                momentum=self.momentum, name=f"{name}{convolution}_batch_norm"
            )(x)
            x = tf.keras.layers.Activation("relu", name=f"{name}{convolution}_relu")(x)

        x = tf.keras.layers.Add(name=f"{name}_skip_add")([x, shortcut_add])
        return tf.keras.layers.ReLU(name=f"{name}_out")(x)


@factory
class StrongBaselineNetBANFactory(StrongBaselineNetFactory):
    model_name = Field("baseline_ban")
    input_quantizer = "ste_sign"
    kernel_quantizer = None
    kernel_constraint = None

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(1e-5)


@factory
class StrongBaselineNetBNNFactory(StrongBaselineNetFactory):
    model_name = Field("baseline_bnn")
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"


@factory
class RealToBinNetFPFactory(RealToBinNetFactory):
    model_name = Field("r2b_fp")
    kernel_quantizer = None
    kernel_constraint = None

    @property
    def input_quantizer(self):
        return tf.keras.layers.Activation("tanh")

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(1e-5)


@factory
class RealToBinNetBANFactory(RealToBinNetFactory):
    model_name = Field("r2b_ban")
    input_quantizer = "ste_sign"
    kernel_quantizer = None
    kernel_constraint = None

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(1e-5)


@factory
class RealToBinNetBNNFactory(RealToBinNetFactory):
    model_name = Field("r2b_bnn")
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"


@factory
class ResNet18FPFactory(ResNet18Factory):
    model_name = Field("resnet_fp")
    input_quantizer = None
    kernel_quantizer = None
    kernel_constraint = None

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(1e-5)


def RealToBinaryNet(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the BNN version of the Real-to-Binary network from Martinez et. al.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    r2b-v0.1.0/r2b.json
    ```
    ```summary
    literature.RealToBinaryNet
    ```
    ```plot-altair
    /plots/r2b_final_stage.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 65.01 %        | 85.72 %        | 11 995 624 | 5.13 MB |

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
        - [Training binary neural networks with real-to-binary
            convolutions](https://openreview.net/forum?id=BJg4NgBKvH)
    """
    return RealToBinNetBNNFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
