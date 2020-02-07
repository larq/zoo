from typing import Tuple

import larq as lq
import tensorflow as tf
from zookeeper import ComponentField, Field, factory, task

from larq_zoo import utils
from larq_zoo.model_factory import ModelFactory
from larq_zoo.train import TrainLarqZooModel


@factory
class BinaryAlexNetFactory(ModelFactory):
    """
    Implementation of ["Binarized Neural Networks"](https://papers.nips.cc/paper/6573-binarized-neural-networks)
    by Hubara et al., NIPS, 2016.
    """

    inflation_ratio: int = Field(1)

    input_quantizer = Field("ste_sign")
    kernel_quantizer = Field("ste_sign")
    kernel_constraint = Field("weight_clip")

    @Field
    def num_classes(self) -> int:
        if self.dataset is None:
            raise TypeError("Must override either `dataset` or `num_classes`.")
        return self.dataset.num_classes

    def conv_block(
        self,
        x: tf.Tensor,
        features: int,
        kernel_size: Tuple[int, int],
        strides: int = 1,
        pool: bool = False,
        first_layer: bool = False,
        no_inflation: bool = False,
    ):
        x = lq.layers.QuantConv2D(
            features * (1 if no_inflation else self.inflation_ratio),
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=None if first_layer else self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        if pool:
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)
        return x

    def dense_block(self, x: tf.Tensor, units: int) -> tf.Tensor:
        x = lq.layers.QuantDense(
            units,
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        return tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)

    def build(self) -> tf.keras.models.Model:
        image_input = self.image_input

        # Feature extractor
        out = self.conv_block(
            image_input,
            features=64,
            kernel_size=11,
            strides=4,
            pool=True,
            first_layer=True,
        )
        out = self.conv_block(out, features=192, kernel_size=5, pool=True)
        out = self.conv_block(out, features=384, kernel_size=3)
        out = self.conv_block(out, features=384, kernel_size=3)
        out = self.conv_block(
            out, features=256, kernel_size=3, pool=True, no_inflation=True
        )

        # Classifier
        if self.include_top:
            out = tf.keras.layers.Flatten()(out)
            out = self.dense_block(out, units=4096)
            out = self.dense_block(out, units=4096)
            out = self.dense_block(out, self.num_classes)
            out = tf.keras.layers.Activation("softmax")(out)

        model = tf.keras.models.Model(
            inputs=image_input, outputs=out, name="binary_alexnet"
        )

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="binary_alexnet",
                    version="v0.2.0",
                    file="binary_alexnet_weights.h5",
                    file_hash="0f8d3f6c1073ef993e2e99a38f8e661e5efe385085b2a84b43a7f2af8500a3d3",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="binary_alexnet",
                    version="v0.2.0",
                    file="binary_alexnet_weights_notop.h5",
                    file_hash="1c7e2ef156edd8e7615e75a3b8929f9025279a948d1911824c2f5a798042475e",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)

        return model


def BinaryAlexNet(
    *,  # Keyword arguments only
    input_shape=None,
    input_tensor=None,
    weights="imagenet",
    include_top=True,
    num_classes=1000,
) -> tf.keras.models.Model:
    """Instantiates the BinaryAlexNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    ```netron
    binary_alexnet-v0.2.0/binary_alexnet.json
    ```
    ```plot-altair
    /plots/binary_alexnet.vg.json
    ```
    # Arguments
    input_shape: optional shape tuple, only to be specified if `include_top` is False,
        otherwise the input shape has to be `(224, 224, 3)`.
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
    """
    return BinaryAlexNetFactory(
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        num_classes=num_classes,
    ).build()


@task
class TrainBinaryAlexNet(TrainLarqZooModel):
    model = ComponentField(BinaryAlexNetFactory)

    batch_size: int = Field(512)
    epochs: int = Field(150)

    def learning_rate_schedule(self, epoch):
        return 1e-2 * 0.5 ** (epoch // 10)

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(self.learning_rate_schedule(0))
    )
