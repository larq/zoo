from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from larq_zoo import utils


@registry.register_model
def birealnet(args, input_shape, num_classes, input_tensor=None, include_top=True):
    def residual_block(x, double_filters=False, filters=None):
        assert not (double_filters and filters)

        # compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        shortcut = x
        if in_filters != out_filters:
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
            shortcut = tf.keras.layers.Conv2D(
                out_filters,
                1,
                kernel_initializer=args.kernel_initializer,
                use_bias=False,
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)

        x = lq.layers.QuantConv2D(
            out_filters,
            3,
            strides=1 if out_filters == in_filters else 2,
            padding="same",
            input_quantizer=args.input_quantizer,
            kernel_quantizer=args.kernel_quantizer,
            kernel_initializer=args.kernel_initializer,
            kernel_constraint=args.kernel_constraint,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        return tf.keras.layers.add([x, shortcut])

    img_input = utils.get_input_layer(input_shape, input_tensor)

    # layer 1
    out = tf.keras.layers.Conv2D(
        args.filters,
        7,
        strides=2,
        kernel_initializer=args.kernel_initializer,
        padding="same",
        use_bias=False,
    )(img_input)
    out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)

    # layer 2
    out = residual_block(out, filters=args.filters)

    # layer 3 - 5
    for _ in range(3):
        out = residual_block(out)

    # layer 6 - 17
    for _ in range(3):
        out = residual_block(out, double_filters=True)
        for _ in range(3):
            out = residual_block(out)

    # layer 18
    if include_top:
        out = tf.keras.layers.GlobalAvgPool2D()(out)
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out, name="birealnet18")


@registry.register_hparams(birealnet)
class default(HParams):
    filters = 64
    learning_rate = 5e-3
    decay_schedule = "linear"
    epochs = 300
    batch_size = 512
    input_quantizer = "approx_sign"
    kernel_quantizer = "magnitude_aware_sign"
    kernel_constraint = "weight_clip"
    kernel_initializer = "glorot_normal"

    @property
    def optimizer(self):
        if self.decay_schedule == "linear_cosine":
            lr = tf.keras.experimental.LinearCosineDecay(self.learning_rate, 750684)
        elif self.decay_schedule == "linear":
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                self.learning_rate, 750684, end_learning_rate=0, power=1.0
            )
        else:
            lr = self.learning_rate
        return tf.keras.optimizers.Adam(lr)


def BiRealNet(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the Bi-Real Net architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    birealnet-v0.3.0/birealnet.json
    ```
    ```plot-altair
    /plots/birealnet.vg.json
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
    - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
      Representational Capability and Advanced Training
      Algorithm](https://arxiv.org/abs/1808.00278)
    """
    input_shape = utils.validate_input(input_shape, weights, include_top, classes)

    model = birealnet(
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
                model="birealnet",
                version="v0.3.0",
                file="birealnet_weights.h5",
                file_hash="6e6efac1584fcd60dd024198c87f42eb53b5ec719a5ca1f527e1fe7e8b997117",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="birealnet",
                version="v0.3.0",
                file="birealnet_weights_notop.h5",
                file_hash="5148b61c0c2a1094bdef811f68bf4957d5ba5f83ad26437b7a4a6855441ab46b",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model
