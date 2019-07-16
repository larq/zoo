import tensorflow as tf
import larq as lq
from larq_zoo import utils
from zookeeper import registry, HParams


@registry.register_model
def binary_alexnet(hparams, dataset, input_tensor=None, include_top=True):
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

    def conv_block(
        x,
        features,
        kernel_size,
        strides=1,
        pool=False,
        first_layer=False,
        no_inflation=False,
    ):
        x = lq.layers.QuantConv2D(
            features * (1 if no_inflation else hparams.inflation_ratio),
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=None if first_layer else "ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
        )(x)
        if pool:
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)
        return x

    def dense_block(x, units):
        x = lq.layers.QuantDense(units, **kwhparams)(x)
        x = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)
        return x

    # get input
    img_input = utils.get_input_layer(dataset.input_shape, input_tensor)

    # feature extractor
    out = conv_block(
        img_input, features=64, kernel_size=11, strides=4, pool=True, first_layer=True
    )
    out = conv_block(out, features=192, kernel_size=5, pool=True)
    out = conv_block(out, features=384, kernel_size=3)
    out = conv_block(out, features=384, kernel_size=3)
    out = conv_block(out, features=256, kernel_size=3, pool=True, no_inflation=True)

    # classifier
    if include_top:
        out = tf.keras.layers.Flatten()(out)
        out = dense_block(out, units=4096)
        out = dense_block(out, units=4096)
        out = dense_block(out, dataset.num_classes)
        out = tf.keras.layers.Activation("softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)


@registry.register_hparams(binary_alexnet)
class default(HParams):
    epochs = 150
    inflation_ratio = 1
    batch_size = 512
    learning_rate = 0.01
    lr_decay_stepsize = 10

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
        # download appropriate file
        if include_top:
            weights_path = utils.download_pretrained_model(
                model="binary_alexnet",
                version="v0.1.0",
                file="binary_alexnet_weights.h5",
                file_hash="4dd4ddf6af0c26a0312ae1109191c1db59707ba0c03a025e6b71035d1f7057a2",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="binary_alexnet",
                version="v0.1.0",
                file="binary_alexnet_weights_notop.h5",
                file_hash="2138b34787df6fee300e584df35de3bf4aefa7dd2788211d2b0d30ae7a9284cc",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
