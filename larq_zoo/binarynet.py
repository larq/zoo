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
                11,
                strides=4,
                padding="same",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                use_bias=False,
                input_shape=dataset.input_shape,
            ),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(hparams.filters * 3, 5, padding="same", **kwargs),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(6 * hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(4 * hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(4 * hparams.filters, 3, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
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
    def lr_schedule(epoch):
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

    return HParams(
        optimizer=tf.keras.optimizers.Adam(5e-3),
        learning_rate_schedule=lr_schedule,
        batch_size=256,
        filters=64,
        dense_units=4096,
    )


class ImagenetDatasetMock:
    input_shape = (224, 224, 3)
    num_classes = 1000


def BinaryAlexNet():
    return binary_alex_net(default(), ImagenetDatasetMock())
