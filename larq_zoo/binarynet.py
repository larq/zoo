from zoo_keeper import registry, HParams
import larq as lq
import tensorflow as tf
from larq_zoo import utils


@registry.register_model
def binary_alex_net(hparams, dataset, input_tensor=None, include_top=True):
    kwargs = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
    )
    img_input = utils.get_input_layer(dataset.input_shape, input_tensor)

    x = lq.layers.QuantConv2D(
        hparams.filters,
        11,
        strides=4,
        padding="same",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
    )(img_input)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantConv2D(hparams.filters * 3, 5, padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantConv2D(6 * hparams.filters, 3, padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantConv2D(4 * hparams.filters, 3, padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantConv2D(4 * hparams.filters, 3, padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = lq.layers.QuantDense(hparams.dense_units, **kwargs)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = lq.layers.QuantDense(hparams.dense_units, **kwargs)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = lq.layers.QuantDense(dataset.num_classes, **kwargs)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = tf.keras.layers.Activation("softmax", name="predictions")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return tf.keras.models.Model(inputs, x, name="binary_alex_net")


@registry.register_hparams(binary_alex_net)
class default(HParams):
    batch_size = 256
    filters = 64
    dense_units = 4096
    optimizer = tf.keras.optimizers.Adam(5e-3)

    def learning_rate_schedule(epoch):
        if epoch < 20:
            return 5e-3
        elif epoch < 30:
            return 1e-3
        elif epoch < 35:
            return 5e-4
        elif epoch < 40:
            return 1e-4
        else:
            return 1e-5


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

    model = binary_alex_net(
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
