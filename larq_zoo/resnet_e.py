from zookeeper import registry, HParams
from tensorflow import keras
from larq_zoo import utils
import larq as lq


def residual_block(x, args, filters, strides=1):
    downsample = x.get_shape().as_list()[-1] != filters

    if downsample:
        residual = keras.layers.AvgPool2D(pool_size=2, strides=2)(x)
        residual = keras.layers.Conv2D(
            filters, kernel_size=1, use_bias=False, kernel_initializer="glorot_normal"
        )(residual)
        residual = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(residual)
    else:
        residual = x

    x = lq.layers.QuantConv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        input_quantizer=args.quantizer,
        kernel_quantizer=args.quantizer,
        kernel_constraint=args.constraint,
        kernel_initializer="glorot_normal",
        use_bias=False,
        metrics=[],
    )(x)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    return keras.layers.add([x, residual])


@registry.register_model
def resnet_e(args, dataset, input_tensor=None, include_top=True):
    input = utils.get_input_layer(dataset.input_shape, input_tensor)

    if dataset.input_shape[0] and dataset.input_shape[0] < 50:
        x = keras.layers.Conv2D(
            args.initial_filters,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )(input)
    else:
        x = keras.layers.Conv2D(
            args.initial_filters,
            kernel_size=7,
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )(input)

        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D(3, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    for block, (layers, filters) in enumerate(zip(*args.spec)):
        # this tricks adds shortcut connections between original resnet blocks
        # we multiple number of blocks by 2, but add only one layer instead of two in each block
        for layer in range(layers * 2):
            strides = 1 if block == 0 or layer != 0 else 2
            x = residual_block(x, args, filters, strides=strides)

    if include_top:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.GlobalAvgPool2D()(x)
        x = keras.layers.Dense(
            dataset.num_classes,
            activation="softmax",
            kernel_initializer="glorot_normal",
        )(x)
    return keras.Model(inputs=input, outputs=x)


@registry.register_hparams(resnet_e)
class default(HParams):
    epochs = 120
    batch_size = 1024
    num_layers = 18
    learning_rate = 0.004
    learning_factor = 0.3
    learning_steps = [70, 90, 110]
    initial_filters = 64
    quantizer = lq.quantizers.SteSign(clip_value=1.25)
    constraint = lq.constraints.WeightClip(clip_value=1.25)

    def learning_rate_schedule(self, epoch):
        lr = self.learning_rate
        for step in self.learning_steps:
            if epoch < step:
                return lr
            lr *= self.learning_factor
        return lr

    @property
    def optimizer(self):
        return keras.optimizers.Adam(self.learning_rate, epsilon=1e-8)

    @property
    def spec(self):
        spec = {
            18: ([2, 2, 2, 2], [64, 128, 256, 512]),
            34: ([3, 4, 6, 3], [64, 128, 256, 512]),
            50: ([3, 4, 6, 3], [256, 512, 1024, 2048]),
            101: ([3, 4, 23, 3], [256, 512, 1024, 2048]),
            152: ([3, 8, 36, 3], [256, 512, 1024, 2048]),
        }
        try:
            return spec[self.num_layers]
        except:
            raise ValueError(f"Only specs for layers {list(self.spec.keys())} defined.")


def ResNetE18(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the ResNetE 18 architecture.

    Optionally loads weights pre-trained on ImageNet.

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
    - [Back to Simplicity:
      How to Train Accurate BNNs from Scratch?](https://arxiv.org/abs/1906.08637)
    """
    input_shape = utils.validate_input(input_shape, weights, include_top, classes)

    model = resnet_e(
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
                model="resnet_e",
                version="v0.1.0",
                file="resnet_e_18_weights.h5",
                file_hash="bde4a64d42c164a7b10a28debbe1ad5b287c499bc0247ecb00449e6e89f3bf5b",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="resnet_e",
                version="v0.1.0",
                file="resnet_e_18_weights_notop.h5",
                file_hash="14cb037e47d223827a8d09db88ec73d60e4153a4464dca847e5ae1a155e7f525",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model
