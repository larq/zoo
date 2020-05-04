from typing import Optional, Sequence

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


# A type alias only for type-checking.
class BinaryDenseNet(tf.keras.models.Model):
    pass


################
# Base factory #
################


class BinaryDenseNetFactory(ModelFactory):
    """Implementation of [BinaryDenseNet](https://arxiv.org/abs/1906.08637)"""

    @property
    def input_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.3)

    @property
    def kernel_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.3)

    @property
    def kernel_constraint(self):
        return lq.constraints.WeightClip(clip_value=1.3)

    initial_filters: int = Field(64)
    growth_rate: int = Field(64)

    # These are not `Fields`, as they should not be configurable, but set in the
    # various concrete subclasses.
    name: str
    reduction: Sequence[float]
    dilation_rate: Sequence[int]
    layers: Sequence[int]
    imagenet_weights_path: str
    imagenet_no_top_weights_path: str

    def densely_connected_block(self, x: tf.Tensor, dilation_rate: int = 1):
        y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        y = lq.layers.QuantConv2D(
            filters=self.growth_rate,
            kernel_size=3,
            dilation_rate=dilation_rate,
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer="glorot_normal",
            kernel_constraint=self.kernel_constraint,
            padding="same",
            use_bias=False,
        )(y)
        return tf.keras.layers.concatenate([x, y])

    def build(self) -> BinaryDenseNet:
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

        for block, layers_per_block in enumerate(self.layers):
            for _ in range(layers_per_block):
                x = self.densely_connected_block(x, self.dilation_rate[block])

            if block < len(self.layers) - 1:
                x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
                if self.dilation_rate[block + 1] == 1:
                    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
                x = tf.keras.layers.Activation("relu")(x)
                x = tf.keras.layers.Conv2D(
                    round(x.shape.as_list()[-1] // self.reduction[block] / 32) * 32,
                    kernel_size=1,
                    kernel_initializer="he_normal",
                    use_bias=False,
                )(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = tf.keras.layers.Activation("relu")(x)

        if self.include_top:
            x = utils.global_pool(x)
            x = tf.keras.layers.Dense(self.num_classes, kernel_initializer="he_normal")(
                x
            )
            x = tf.keras.layers.Activation("softmax", dtype="float32")(x)

        model = BinaryDenseNet(inputs=self.image_input, outputs=x, name=self.name)

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
class BinaryDenseNet28Factory(BinaryDenseNetFactory):
    name = "binary_densenet28"
    reduction = (2.7, 2.7, 2.2)
    dilation_rate = (1, 1, 1, 1)
    layers = (6, 6, 6, 5)

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.0",
            file="binary_densenet_28_weights.h5",
            file_hash="21fe3ca03eed244df9c41a2219876fcf03e73800932ec96a3e2a76af4747ac53",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.0",
            file="binary_densenet_28_weights_notop.h5",
            file_hash="a376df1e41772c4427edd1856072b934a89bf293bf911438bf6f751a9b2a28f5",
        )


@factory
class BinaryDenseNet37Factory(BinaryDenseNetFactory):
    name = "binary_densenet37"
    reduction = (3.3, 3.3, 4)
    dilation_rate = (1, 1, 1, 1)
    layers = (6, 8, 12, 6)

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.0",
            file="binary_densenet_37_weights.h5",
            file_hash="8056a5d52c3ed86a934893987d09a06f59a5166aa9bddcaedb050f111d0a7d76",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.0",
            file="binary_densenet_37_weights_notop.h5",
            file_hash="4e12bca9fd27580a5b833241c4eb35d6cc332878c406048e6ca8dbbc78d59175",
        )


@factory
class BinaryDenseNet37DilatedFactory(BinaryDenseNetFactory):
    name = "binary_densenet37_dilated"
    reduction = (3.3, 3.3, 4)
    dilation_rate = (1, 1, 2, 4)
    layers = (6, 8, 12, 6)

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.0",
            file="binary_densenet_37_dilated_weights.h5",
            file_hash="15c1bcd79b8dc22971382fbf79acf364a3f51049d0e584a11533e6fdbb7363d3",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.1",
            file="binary_densenet_37_dilated_weights_notop.h5",
            file_hash="8b31fbfdc8de08a46c6adcda1ced48ace0a2ff0ce45a05c72b2acc27901dd88b",
        )


@factory
class BinaryDenseNet45Factory(BinaryDenseNet28Factory):
    name = "binary_densenet45"
    reduction = (2.7, 3.3, 4)
    dilation_rate = (1, 1, 1, 1)
    layers = (6, 12, 14, 8)

    @property
    def imagenet_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.0",
            file="binary_densenet_45_weights.h5",
            file_hash="d00a0d26fbd2dba1bfba8c0306c770f3aeea5c370e99f963bb239bd916f72c37",
        )

    @property
    def imagenet_no_top_weights_path(self):
        return utils.download_pretrained_model(
            model="binary_densenet",
            version="v0.1.0",
            file="binary_densenet_45_weights_notop.h5",
            file_hash="e72d5cc6b0afe4612f8be7b1f9bb48a53ba2c8468b57bf1266d2900c99fd2adf",
        )


#########################
# Functional interfaces #
#########################


def BinaryDenseNet28(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the BinaryDenseNet 28 architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    binary_densenet-v0.1.0/binary_densenet_28.json
    ```
    ```summary
    literature.BinaryDenseNet28
    ```
    ```plot-altair
    /plots/densenet_28.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 60.91 %        | 82.83 %        | 5 150 504  | 4.12 MB |

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
    return BinaryDenseNet28Factory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


def BinaryDenseNet37(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the BinaryDenseNet 37 architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    binary_densenet-v0.1.0/binary_densenet_37.json
    ```
    ```summary
    literature.BinaryDenseNet37
    ```
    ```plot-altair
    /plots/densenet_37.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 62.89 %        | 84.19 %        | 8 734 120  | 5.25 MB |

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
    return BinaryDenseNet37Factory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


def BinaryDenseNet37Dilated(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the BinaryDenseNet 37Dilated architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    binary_densenet-v0.1.0/binary_densenet_37_dilated.json
    ```
    ```summary
    literature.BinaryDenseNet37Dilated
    ```
    ```plot-altair
    /plots/densenet_37_dilated.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 64.34 %        | 85.15 %        | 8 734 120  | 5.25 MB |

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
    return BinaryDenseNet37DilatedFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


def BinaryDenseNet45(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[tf.Tensor] = None,
    weights: Optional[str] = "imagenet",
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the BinaryDenseNet 45 architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    binary_densenet-v0.1.0/binary_densenet_45.json
    ```
    ```summary
    literature.BinaryDenseNet45
    ```
    ```plot-altair
    /plots/densenet_45.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 64.59 %        | 85.21 %        | 13 939 240 | 7.54 MB |

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
    return BinaryDenseNet45Factory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()
