from typing import Optional, Sequence, Tuple, Union

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory

################
# Base factory #
################


class MeliusNetFactory(ModelFactory):
    # Overall architecture configuration. These are not `Fields`, as they should
    # not be configurable, but set in the various concrete subclasses.
    num_blocks: Sequence[int]
    transition_features: Sequence[int]
    name: str

    # Some default layer arguments.
    batch_norm_momentum: float = Field(0.9)
    kernel_initializer: Optional[Union[str, tf.keras.initializers.Initializer]] = Field(
        "glorot_normal"
    )
    input_quantizer = Field(lambda: lq.quantizers.SteSign(1.3))
    kernel_quantizer = Field(lambda: lq.quantizers.SteSign(1.3))
    kernel_constraint = Field(lambda: lq.constraints.WeightClip(1.3))

    def pool(self, x: tf.Tensor, name: str) -> tf.Tensor:
        return tf.keras.layers.MaxPool2D(2, strides=2, padding="same")(x)

    def norm(self, x: tf.Tensor, name: str) -> tf.Tensor:
        return tf.keras.layers.BatchNormalization(
            momentum=self.batch_norm_momentum, epsilon=1e-5
        )(x)

    def act(self, x: tf.Tensor, name: str) -> tf.Tensor:
        return tf.keras.layers.Activation("relu")(x)

    def quant_conv(
        self,
        x: tf.Tensor,
        filters: int,
        name: str,
        kernel: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = 1,
    ) -> tf.Tensor:
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

    def group_conv(
        self,
        x: tf.Tensor,
        filters: int,
        kernel: Union[int, Tuple[int, int]],
        groups: int,
    ) -> tf.Tensor:
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

    def group_stem(self, x: tf.Tensor) -> tf.Tensor:
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

    def dense_block(self, x: tf.Tensor) -> tf.Tensor:
        w = x
        w = self.norm(w)
        w = self.quant_conv(w, 64, 3)
        return tf.concat([x, w], axis=-1)

    def improvement_block(self, x: tf.Tensor) -> tf.Tensor:
        w = x
        w = self.norm(w)
        w = self.quant_conv(w, 64, 3)
        f_in = int(x.shape[-1])
        return x + tf.pad(w, [[0, 0], [0, 0], [0, 0], [f_in - 64, 0]])

    def transition_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        x = self.norm(x)
        x = self.pool(x)
        x = self.act(x)
        return tf.keras.layers.Conv2D(
            filters, 1, use_bias=False, kernel_initializer=self.kernel_initializer
        )(x)

    def block(self, x: tf.Tensor) -> tf.Tensor:
        x = self.dense_block(x)
        return self.improvement_block(x)

    def build(self) -> tf.keras.models.Model:
        x = self.image_input
        x = self.group_stem(x)
        for i, (n, f) in enumerate(zip(self.num_blocks, self.transition_features)):
            for j in range(n):
                x = self.block(x)
            if f:
                x = self.transition_block(x, f)

        x = self.norm(x)
        x = self.act(x)

        if self.include_top:
            x = utils.global_pool(x)
            x = tf.keras.layers.Dense(
                self.num_classes, kernel_initializer=self.kernel_initializer
            )(x)
            x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        model = tf.keras.models.Model(
            inputs=self.image_input, outputs=x, name=self.name
        )

        if self.weights == "imagenet":
            if self.include_top:
                weights_path = self.imagenet_weights_path
            else:
                weights_path = self.imagenet_no_top_weights_path
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)

        return model


######################
# Concrete factories #
######################


@factory
class MeliusNet22Factory(MeliusNetFactory):
    num_blocks = (4, 5, 4, 4)
    transition_features = (160, 224, 256, None)
    name = "meliusnet22"

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="meliusnet22",
            version="v0.1.0",
            file="meliusnet22_weights.h5",
            file_hash="bb8dda20642508bbe5e0ff95012fec450103c4b23989f4c9c9d853d67b6ff806",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="meliusnet22",
            version="v0.1.0",
            file="meliusnet22_weights_notop.h5",
            file_hash="9ca867806bff0c2995ff5f1ad085d1627c8dabf12bffbf0bea86eb39ab3cf724",
        )


#########################
# Functional interfaces #
#########################


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
    ```summary
    literature.MeliusNet22
    ```
    ```plot-altair
    /plots/meliusnet22.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory   |
    | -------------- | -------------- | ---------- | -------- |
    | 62.4 %         | 83.9 %         | 6 944 584  | 3.88 MiB |

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
    - [MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?](https://arxiv.org/abs/2001.05936)
    """
    return MeliusNet22Factory(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
    ).build()
