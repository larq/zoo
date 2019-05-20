from larq_flock import registry, HParams
import larq as lq
import tensorflow as tf
from larq_zoo import layers, optimizers


@registry.register_model
def birealnet(args, dataset):
    def residual_block(x, double_filters=False, filters=None):
        assert not (double_filters and filters)

        # figure out dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        reduce_size = in_filters != out_filters

        # shortcut
        shortcut = x
        if reduce_size:
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
        if in_filters != out_filters:
            shortcut = tf.keras.layers.Conv2D(
                out_filters,
                1,
                kernel_initializer=args.kernel_initializer,
                use_bias=False,
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.9)(shortcut)

        # actual convolution
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
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)

        return tf.keras.layers.add([x, shortcut])

    inp = tf.keras.layers.Input(shape=dataset.input_shape)

    # layer 1
    out = tf.keras.layers.Conv2D(
        args.filters,
        7,
        strides=2,
        kernel_initializer=args.kernel_initializer,
        padding="same",
        use_bias=False,
    )(inp)
    out = tf.keras.layers.BatchNormalization(momentum=0.9)(out)
    # out = normalization.GroupNormalization(groups=32)(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)

    # layer 2 - 5
    out = residual_block(out, filters=args.filters)
    for _ in range(1, 5):
        out = residual_block(out)

    # layer 6 - 17
    for i in range(1, 4):
        out = residual_block(out, double_filters=True)
        for _ in range(1, 4):
            out = residual_block(out)

    # layer 18
    out = tf.keras.layers.GlobalAvgPool2D()(out)
    out = tf.keras.layers.Dense(dataset.num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs=inp, outputs=out)


@registry.register_hparams(birealnet)
def default():
    initial_lr = 1e-3
    return HParams(
        learning_rate_schedule=lambda epoch: initial_lr
        * 0.1 ** (epoch // 55)
        * 0.1 ** (epoch // 65)
        * 0.1 ** (epoch // 70),
        filters=64,
        optimizer=tf.keras.optimizers.Adam(initial_lr),
        batch_size=256,
        input_quantizer="approx_sign",
        kernel_quantizer="magnitude_aware_sign",
        kernel_constraint="weight_clip",
        kernel_initializer="glorot_normal",
    )


@registry.register_hparams(birealnet)
def bop():
    return HParams(
        filters=64,
        optimizer=optimizers.Bop(tf.keras.optimizers.Adam()),
        batch_size=256,
        input_quantizer="approx_sign",
        kernel_quantizer=None,
        kernel_constraint=None,
        kernel_initializer="glorot_normal",
    )


@registry.register_hparams(birealnet)
def latent_free():
    initial_lr = 0.01
    return HParams(
        filters=64,
        optimizer=tf.keras.optimizers.SGD(initial_lr, momentum=0.99),
        learning_rate_schedule=lambda epoch: initial_lr * 0.95 ** epoch,
        batch_size=256,
        input_quantizer="approx_sign",
        kernel_quantizer=None,
        kernel_constraint=layers.LatentFree(1e-6),
        kernel_initializer="glorot_normal",
    )
