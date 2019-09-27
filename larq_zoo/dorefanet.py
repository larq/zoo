import tensorflow as tf
import larq as lq
from larq_zoo import utils
from zookeeper import registry, HParams


@registry.register_model
def dorefa_net(hparams, input_shape, num_classes, input_tensor=None, include_top=True):
    def conv_block(x, filters, kernel_size, strides=1, pool=False, padding="same"):
        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=hparams.input_quantizer,
            kernel_quantizer=hparams.kernel_quantizer,
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9, epsilon=1e-4)(
            x
        )
        if pool:
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding=padding)(x)
        return x

    def dense_block(x, units):

        x = lq.layers.QuantDense(
            units,
            input_quantizer=hparams.input_quantizer,
            kernel_quantizer=hparams.kernel_quantizer,
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9, epsilon=1e-4)(
            x
        )
        return x

    # get input
    img_input = utils.get_input_layer(input_shape, input_tensor)

    # feature extractor
    out = tf.keras.layers.Conv2D(
        96, kernel_size=12, strides=4, padding="valid", use_bias=True
    )(img_input)
    out = conv_block(out, features=256, kernel_size=5, pool=True)
    out = conv_block(out, features=384, kernel_size=3, pool=True)
    out = conv_block(out, features=384, kernel_size=3)
    out = conv_block(out, features=256, kernel_size=3, padding="valid", pool=True)

    # classifier
    if include_top:
        out = tf.keras.layers.Flatten()(out)
        out = dense_block(out, units=4096)
        out = dense_block(out, units=4096)
        out = tf.keras.layers.Activation("clip_by_value_activation")(out)
        out = tf.keras.layers.Dense(num_classes, use_bias=True)(out)
        out = tf.keras.layers.Activation("softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)


@lq.utils.register_keras_custom_object
@lq.utils.set_precision(1)
def magnitude_aware_sign_unclipped(x):
    r"""
    Scaled sign function with identity pseudo-gradient as used for the
    weights in the DoReFa paper. The Scale factor is calculated per layer.
    """
    scale_factor = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

    @tf.custom_gradient
    def _magnitude_aware_sign(x):
        return lq.math.sign(x) * scale_factor, lambda dy: dy

    return _magnitude_aware_sign(x)


@lq.utils.register_keras_custom_object
def clip_by_value_activation(x):
    return tf.clip_by_value(x, 0, 1)


@registry.register_hparams(dorefa_net)
class default(HParams):
    epochs = 90
    batch_size = 256
    learning_rate = 0.0002
    decay_start = 60
    decay_step_2 = 75
    fast_decay_start = 82
    activations_k_bit = 2

    @property
    def input_quantizer(self):
        return lq.quantizers.DoReFaQuantizer(k_bit=self.activations_k_bit)

    @property
    def kernel_quantizer(self):
        return magnitude_aware_sign_unclipped

    def learning_rate_schedule(self, epoch):
        if epoch < self.decay_start:
            return self.learning_rate
        elif epoch < self.decay_step_2:
            return 4e-5
        elif epoch < self.fast_decay_start:
            return 8e-6
        else:
            return 8e-6 * 0.1 ** ((epoch - self.fast_decay_start) // 2 + 1)

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-5)


def DoReFaNet(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the DoReFa-net architecture.
    Optionally loads weights pre-trained on ImageNet.
    ```netron
    dorefanet-v0.1.0/dorefanet.json
    ```
    ```plot-altair
    /plots/dorefanet.vg.json
    ```
    # Arguments
    include_top: whether to include the fully-connected layer at the top of the network.
    weights: one of `None` (random initialization), "imagenet" (pre-training on
        ImageNet), or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
        image input for the model.
    input_shape: optional shape tuple, only to be specified if `include_top` is False,
        otherwise the input shape has to be `(224, 224, 3)`.
        It should have exactly 3 inputs channels.
    classes: optional number of classes to classify images into, only to be specified
        if `include_top` is True, and if no `weights` argument is specified.
    # Returns
    A Keras model instance.
    # Raises
    ValueError: in case of invalid argument for `weights`, or invalid input shape.
    # References
    - [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low
    Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
    """
    input_shape = utils.validate_input(input_shape, weights, include_top, classes)

    model = dorefa_net(
        default(),
        input_shape,
        classes,
        input_tensor=input_tensor,
        include_top=include_top,
    )

    # Load weights.
    if weights == "imagenet":
        # download appropriate file
        if include_top:
            weights_path = utils.download_pretrained_model(
                model="dorefanet",
                version="v0.1.0",
                file="dorefanet_weights.h5",
                file_hash="645d7839d574faa3eeeca28f3115773d75da3ab67ff6876b4de12d10245ecf6a",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="dorefanet",
                version="v0.1.0",
                file="dorefanet_weights_notop.h5",
                file_hash="679368128e19a2a181bfe06ca3a3dec368b1fd8011d5f42647fbbf5a7f36d45f",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model
