import collections
import tensorflow as tf
import larq as lq
from larq_zoo import utils
from zookeeper import registry, HParams


def create_model(nested_list):
    """Utility function to build a model from a nested list of layers."""

    def unpack(l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(
                el, (str, bytes)
            ):
                yield from unpack(el)
            else:
                yield el

    model = tf.keras.models.Sequential()
    for layer in unpack(nested_list):
        model.add(layer)
    return model


@registry.register_model
def binary_alexnet(hparams, dataset):
    """
    Implementation of ["Binarized Neural Networks"](https://papers.nips.cc/paper/6573-binarized-neural-networks)
    by Hubara et al., NIPS, 2016.
    """
    kwhparams = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
    )

    # Generalized definition of the conv block.
    def conv_block(
        features,
        kernel_size,
        strides=1,
        pool=False,
        first_layer=False,
        no_inflation=False,
    ):
        layers = []
        conv_kwargs = {"input_shape": dataset.input_shape} if first_layer else {}
        layers.append(
            lq.layers.QuantConv2D(
                features * (1 if no_inflation else hparams.inflation_ratio),
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                input_quantizer=None if first_layer else "ste_sign",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                use_bias=False,
                **conv_kwargs
            )
        )
        if pool:
            layers.append(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
        layers.append(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
        return layers

    # Generalized definition of a fully connected block.
    def fc_block(units):
        return [
            lq.layers.QuantDense(units, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False, momentum=0.9),
        ]

    return create_model(
        [
            conv_block(64, 11, strides=4, pool=True, first_layer=True),
            conv_block(192, 5, pool=True),
            conv_block(384, 3),
            conv_block(384, 3),
            conv_block(256, 3, pool=True, no_inflation=True),
            tf.keras.layers.Flatten(),
            fc_block(4096),
            fc_block(4096),
            fc_block(dataset.num_classes),
            tf.keras.layers.Activation("softmax"),
        ]
    )


@registry.register_hparams(binary_alexnet)
class default(HParams):
    epochs = 100
    inflation_ratio = 1
    batch_size = 512
    learning_rate = 0.01
    lr_decay_stepsize = 10
    xavier_scaling = False

    def learning_rate_schedule(self, epoch):
        return self.learning_rate * 0.5 ** (epoch // self.lr_decay_stepsize)

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)


def BinaryAlexNet(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the BinaryAlexNet architecture.

    Optionally loads weights pre-trained on ImageNet.

    # Arguments
    include_top: whether to include the fully-connected layer at the top of the network.
    weights: one of `None` (random initialization), "imagenet" (pre-training on
        ImageNet), or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
        image input for the model.
    input_shape: optional shape tuple, only to be specified if `include_top` is False
        (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data
        format) or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
    classes: optional number of classes to classify images into, only to be specified
        if `include_top` is True, and if no `weights` argument is specified.

    # Returns
    A Keras model instance.

    # Raises
    ValueError: in case of invalid argument for `weights`, or invalid input shape.
    """
    input_shape = utils.validate_input(input_shape, weights, include_top, classes)

    model = binary_alexnet(
        default(),
        utils.ImagenetDataset(input_shape, classes),
        input_tensor=input_tensor,
        include_top=include_top,
    )

    # Load weights.
    if weights == "imagenet":
        raise NotImplementedError()
        # if include_top:
        #     weights_path = tf.keras.utils.get_file(
        #         "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
        #         WEIGHTS_PATH,
        #         cache_subdir="models",
        #         file_hash="64373286793e3c8b2b4e3219cbf3544b",
        #     )
        # else:
        #     weights_path = tf.keras.utils.get_file(
        #         "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
        #         WEIGHTS_PATH_NO_TOP,
        #         cache_subdir="models",
        #         file_hash="6d6bbae143d832006294945121d1f1fc",
        #     )
        # model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model
