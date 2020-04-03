from typing import Optional, Sequence, Union

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from core import utils
from core.model_factory import ModelFactory


@factory
class MeliusNetFactory(ModelFactory):
    weights: Optional[str] = Field(None)

    # overall architecture configuration
    num_blocks: Sequence[int]
    transition_features: Sequence[int]
    name: str

    # Some default layer arguments.
    batch_norm_momentum: float = Field(0.9)
    kernel_initializer: Optional[Union[str, tf.keras.initializers.Initializer]] = Field(
        "glorot_normal"
    )
    input_quantizer = Field(lq.quantizers.SteSign(1.3))
    kernel_quantizer = Field(lq.quantizers.SteSign(1.3))
    kernel_constraint = Field("weight_clip")

    def pool(self, x):
        return tf.keras.layers.MaxPool2D(2, strides=2, padding="same")(x)

    def norm(self, x):
        return tf.keras.layers.BatchNormalization(
            momentum=self.batch_norm_momentum, epsilon=1e-5
        )(x)

    def act(self, x):
        return tf.keras.layers.Activation("relu")(x)

    def quant_conv(self, x, filters, kernel, strides=1):
        return lq.layers.QuantConv2D(
            filters,
            kernel,
            strides=strides,
            padding="same",
            use_bias=False,
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_initializer=self.kernel_initializer,
        )(x)

    def group_conv(self, x, filters, kernel, groups):
        assert filters % groups == 0
        assert x.shape.as_list()[-1] % groups == 0

        x_split = tf.split(x, groups, axis=-1)

        y_split = [
            tf.keras.layers.Conv2D(
                filters // groups,
                kernel,
                padding="same",
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
            )(split)
            for split in x_split
        ]

        return tf.concat(y_split, axis=-1)

    def group_stem(self, x):
        x = tf.keras.layers.Conv2D(
            32,
            3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(x)
        x = self.norm(x)  # compare Fig 2 and Fig 3
        x = self.act(x)

        x = self.group_conv(x, 32, 3, 4)
        x = self.norm(x)
        x = self.act(x)

        x = self.group_conv(x, 64, 3, 8)
        x = self.norm(x)
        x = self.act(x)

        return self.pool(x)

    def dense_block(self, x):
        w = x
        w = self.norm(w)
        w = self.quant_conv(w, 64, 3)
        return tf.concat([x, w], axis=-1)

    def improvement_block(self, x):
        w = x
        w = self.norm(w)
        w = self.quant_conv(w, 64, 3)
        f_in = int(x.shape[-1])
        return x + tf.pad(w, [[0, 0], [0, 0], [0, 0], [f_in - 64, 0]])

    def transition_block(self, x, filters):
        x = self.norm(x)
        x = self.pool(x)
        x = self.act(x)
        return tf.keras.layers.Conv2D(
            filters, 1, use_bias=False, kernel_initializer=self.kernel_initializer
        )(x)

    def block(self, x):
        x = self.dense_block(x)
        return self.improvement_block(x)

    def build(self) -> tf.keras.models.Model:
        x = self.input_tensor
        x = self.group_stem(x)
        for i, (n, f) in enumerate(zip(self.num_blocks, self.transition_features)):
            for j in range(n):
                x = self.block(x)
            if f:
                x = self.transition_block(x, f)

        x = self.norm(x)
        x = self.act(x)
        x = utils.global_pool(x)
        x = tf.keras.layers.Dense(
            self.num_classes, kernel_initializer=self.kernel_initializer
        )(x)
        x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        model = tf.keras.models.Model(
            inputs=self.input_tensor, outputs=x, name=self.name
        )

        if self.weights:
            weights = self.weights

            if weights.startswith("gs://"):
                model.load_weights(utils.get_gcp_weights(weights))
            else:
                model.load_weights(weights)

        return model


@factory
class MeliusNet22Factory(MeliusNetFactory):
    num_blocks = (4, 5, 4, 4)
    transition_features = (160, 224, 256, None)
    name = "meliusnet22"

    def build(self):
        model = super().build()

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="meliusnet22",
                    version="v0.1.0",
                    file="meliusnet22_weights.h5",
                    file_hash="",  # TODO
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="meliusnet22",
                    version="v0.1.0",
                    file="meliusnet22_weights_notop.h5",
                    file_hash="",  # TODO
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model


def MeliusNet22(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the MeliusNet22 architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    meliusnet22-v0.1.0/meliusnet22.json
    ```
    ```plot-altair
    /plots/meliusnet22.vg.json
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
    - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
      Representational Capability and Advanced Training
      Algorithm](https://arxiv.org/abs/1808.00278)
    """
    return MeliusNet22Factory(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
    ).build()
