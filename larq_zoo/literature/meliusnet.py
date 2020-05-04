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
    name: str = None
    imagenet_weights_path: str
    imagenet_no_top_weights_path: str

    # Some default layer arguments.
    batch_norm_momentum: float = Field(0.9)
    kernel_initializer: Optional[Union[str, tf.keras.initializers.Initializer]] = Field(
        "glorot_normal"
    )

    @property
    def input_quantizer(self):
        return lq.quantizers.SteSign(1.3)

    @property
    def kernel_quantizer(self):
        return lq.quantizers.SteSign(1.3)

    @property
    def kernel_constraint(self):
        return lq.constraints.WeightClip(1.3)

    def pool(self, x: tf.Tensor, name: str = None) -> tf.Tensor:
        return tf.keras.layers.MaxPool2D(2, strides=2, padding="same", name=name)(x)

    def norm(self, x: tf.Tensor, name: str = None) -> tf.Tensor:
        return tf.keras.layers.BatchNormalization(
            momentum=self.batch_norm_momentum, epsilon=1e-5, name=name
        )(x)

    def act(self, x: tf.Tensor, name: str = None) -> tf.Tensor:
        return tf.keras.layers.Activation("relu", name=name)(x)

    def quant_conv(
        self,
        x: tf.Tensor,
        filters: int,
        kernel: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = 1,
        name: str = None,
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
            name=name,
        )(x)

    def group_conv(
        self,
        x: tf.Tensor,
        filters: int,
        kernel: Union[int, Tuple[int, int]],
        groups: int,
        name: str = None,
    ) -> tf.Tensor:
        assert filters % groups == 0
        assert x.shape.as_list()[-1] % groups == 0

        x_split = utils.TFOpLayer(tf.split, groups, axis=-1, name=f"{name}_split")(x)

        y_split = [
            tf.keras.layers.Conv2D(
                filters // groups,
                kernel,
                padding="same",
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                name=f"{name}_conv{i}",
            )(split)
            for i, split in enumerate(x_split)
        ]

        return utils.TFOpLayer(tf.concat, axis=-1, name=f"{name}_concat")(y_split)

    def group_stem(self, x: tf.Tensor, name: str = None) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(
            32,
            3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name=f"{name}_s0_conv",
        )(x)
        x = self.norm(x, name=f"{name}_s0_bn")
        x = self.act(x, name=f"{name}_s0_relu")

        x = self.group_conv(x, 32, 3, 4, name=f"{name}_s1_groupconv")
        x = self.norm(x, name=f"{name}_s1_bn")
        x = self.act(x, name=f"{name}_s1_relu")

        x = self.group_conv(x, 64, 3, 8, name=f"{name}_s2_groupconv")
        x = self.norm(x, name=f"{name}_s2_bn")
        x = self.act(x, name=f"{name}_s2_relu")

        return self.pool(x, name=f"{name}_pool")

    def dense_block(self, x: tf.Tensor, name: str = None) -> tf.Tensor:
        w = x
        w = self.norm(w, name=f"{name}_bn")
        w = self.quant_conv(w, 64, 3, name=f"{name}_binconv")
        return utils.TFOpLayer(tf.concat, axis=-1, name=f"{name}_concat")([x, w])

    def improvement_block(self, x: tf.Tensor, name: str = None) -> tf.Tensor:
        w = x
        w = self.norm(w, name=f"{name}_bn")
        w = self.quant_conv(w, 64, 3, name=f"{name}_binconv")
        f_in = int(x.shape[-1])
        return tf.keras.layers.Lambda(
            lambda x_: x_[0] + tf.pad(x_[1], [[0, 0], [0, 0], [0, 0], [f_in - 64, 0]]),
            name=f"{name}_merge",
        )([x, w])

    def transition_block(
        self, x: tf.Tensor, filters: int, name: str = None
    ) -> tf.Tensor:
        x = self.norm(x, name=f"{name}_bn")
        x = self.pool(x, name=f"{name}_maxpool")
        x = self.act(x, name=f"{name}_relu")
        return tf.keras.layers.Conv2D(
            filters,
            1,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name=f"{name}_pw",
        )(x)

    def block(self, x: tf.Tensor, name: str = None) -> tf.Tensor:
        x = self.dense_block(x, name=f"{name}_dense")
        return self.improvement_block(x, name=f"{name}_improve")

    def build(self) -> tf.keras.models.Model:
        x = self.image_input
        x = self.group_stem(x, name="stem")
        for i, (n, f) in enumerate(zip(self.num_blocks, self.transition_features)):
            for j in range(n):
                x = self.block(x, f"section_{i}_block_{j}")
            if f:
                x = self.transition_block(x, f, f"section_{i}_transition")

        x = self.norm(x, "head_bn")
        x = self.act(x, "head_relu")

        if self.include_top:
            x = utils.global_pool(x, name="head_globalpool")
            x = tf.keras.layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                name="head_dense",
            )(x)
            x = tf.keras.layers.Activation(
                "softmax", dtype="float32", name="head_softmax"
            )(x)

        model = tf.keras.models.Model(
            inputs=self.image_input, outputs=x, name=self.name
        )

        if self.weights == "imagenet":
            model.load_weights(
                self.imagenet_weights_path
                if self.include_top
                else self.imagenet_no_top_weights_path
            )
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
            file_hash="c1ba85e8389ae326009665ec13331e49fc3df4d0f925fa8553e224f7362c18ed",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="meliusnet22",
            version="v0.1.1",
            file="meliusnet22_weights_notop.h5",
            file_hash="abfc5c50049d72a14e44df0c1cb73896ece2a1ab4bf9bb48fede6cc2f5e0b58f",
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
        - [MeliusNet: Can Binary Neural Networks Achieve MobileNet-level
            Accuracy?](https://arxiv.org/abs/2001.05936)
    """
    return MeliusNet22Factory(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
    ).build()
