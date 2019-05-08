from larq_flock import registry, HParams
import larq as lq
import tensorflow as tf


@registry.register_model
def binary_alex_net(hparams, dataset):
    kwargs = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
    )

    return tf.keras.models.Sequential(
        [
            lq.layers.QuantConv2D(
                hparams.filters,
                3,
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                use_bias=False,
                input_shape=dataset.input_shape,
            ),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(2 * hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(2 * hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(4 * hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(4 * hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Flatten(),
            lq.layers.QuantDense(hparams.dense_units, **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantDense(hparams.dense_units, **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantDense(dataset.num_classes, **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Activation("softmax"),
        ]
    )


@registry.register_hparams(binary_alex_net)
def default():
    return HParams(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=1e-3,
        batch_size=256,
        filters=128,
        dense_units=1024,
    )


def BinaryAlexNet():
    raise NotImplementedError()
