from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from larq_zoo import utils


@registry.register_model
def xnornet(hparams, dataset, input_tensor=None, include_top=True):
    kwargs = dict(
        kernel_quantizer=hparams.kernel_quantizer,
        input_quantizer=hparams.input_quantizer,
        kernel_constraint=hparams.kernel_constraint,
        use_bias=hparams.use_bias,
        kernel_regularizer=hparams.kernel_regularizer,
    )
    img_input = utils.get_input_layer(dataset.input_shape, input_tensor)

    x = tf.keras.layers.Conv2D(
        96,
        (11, 11),
        strides=(4, 4),
        padding="same",
        use_bias=hparams.use_bias,
        input_shape=dataset.input_shape,
        kernel_regularizer=hparams.kernel_regularizer,
    )(img_input)

    x = tf.keras.layers.BatchNormalization(
        momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-5
    )(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(
        momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-4
    )(x)
    x = lq.layers.QuantConv2D(256, (5, 5), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(
        momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-4
    )(x)
    x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(
        momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-4
    )(x)
    x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(
        momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-4
    )(x)
    x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(
        momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-4
    )(x)
    x = lq.layers.QuantConv2D(4096, (6, 6), padding="valid", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(
        momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-4
    )(x)

    if include_top:
        # Equivilent to a dense layer
        x = lq.layers.QuantConv2D(
            4096, (1, 1), strides=(1, 1), padding="valid", **kwargs
        )(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=hparams.bn_momentum, scale=hparams.bn_scale, epsilon=1e-3
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            dataset.num_classes,
            use_bias=False,
            kernel_regularizer=hparams.kernel_regularizer,
        )(x)
        x = tf.keras.layers.Activation("softmax")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    return tf.keras.models.Model(inputs, x)


@lq.utils.register_keras_custom_object
def xnor_weight_scale(x):
    """ Clips the weights between -1 and +1 and then
        calculates a scale factor per weight filter. See
        https://arxiv.org/abs/1603.05279 for more details
    """

    x = tf.clip_by_value(x, -1, 1)

    alpha = tf.reduce_mean(tf.abs(x), axis=[0, 1, 2], keepdims=True)

    return alpha * lq.quantizers.ste_sign(x)


@registry.register_hparams(xnornet)
class default(HParams):
    input_quantizer = "ste_sign"
    kernel_quantizer = "xnor_weight_scale"
    kernel_constraint = "weight_clip"
    regularization_quantity = 5e-7
    use_bias = False
    bn_scale = False
    bn_momentum = 0.9
    epochs = 100
    batch_size = 1200
    initial_lr = 0.001

    def learning_rate_schedule(self, epoch):
        epoch_dec_1 = 19
        epoch_dec_2 = 30
        epoch_dec_3 = 44
        epoch_dec_4 = 53
        epoch_dec_5 = 66
        epoch_dec_6 = 76
        epoch_dec_7 = 86
        if epoch < epoch_dec_1:
            return self.initial_lr
        elif epoch < epoch_dec_2:
            return self.initial_lr * 0.5
        elif epoch < epoch_dec_3:
            return self.initial_lr * 0.1
        elif epoch < epoch_dec_4:
            return self.initial_lr * 0.1 * 0.5
        elif epoch < epoch_dec_5:
            return self.initial_lr * 0.01
        elif epoch < epoch_dec_6:
            return self.initial_lr * 0.01 * 0.5
        elif epoch < epoch_dec_7:
            return self.initial_lr * 0.01 * 0.1
        else:
            return self.initial_lr * 0.001 * 0.1

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.initial_lr)


def XNORNet(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the XNOR-Net architecture.

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

    # References
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural
      Networks](https://arxiv.org/abs/1603.05279)
    """
    input_shape = utils.validate_input(input_shape, weights, include_top, classes)

    model = xnornet(
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
                model="xnornet",
                version="v0.1.0",
                file="xnornet_weights.h5",
                file_hash="031d79b139ef2c04125726698e4a90fc6e862114e738ffb9f5b06be635b4e3f0",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="xnornet",
                version="v0.1.0",
                file="xnornet_weights_notop.h5",
                file_hash="7f8ba86da2770ad317b5ab685d6c644128b9ac8ce480444dddaecd2664b6a6b4",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model
