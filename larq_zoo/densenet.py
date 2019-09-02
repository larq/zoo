from zookeeper import registry, HParams
from tensorflow import keras
from larq_zoo import utils
import larq as lq


def densely_connected_block(x, args, dilation_rate=1):
    y = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    y = lq.layers.QuantConv2D(
        filters=args.growth_rate,
        kernel_size=3,
        dilation_rate=dilation_rate,
        input_quantizer=args.quantizer,
        kernel_quantizer=args.quantizer,
        kernel_initializer="glorot_normal",
        kernel_constraint=args.constraint,
        padding="same",
        use_bias=False,
    )(y)
    return keras.layers.concatenate([x, y])


@registry.register_model
def binary_densenet(
    args, input_shape, num_classes, input_tensor=None, include_top=True
):
    input = utils.get_input_layer(input_shape, input_tensor)

    if input_shape[0] and input_shape[0] < 50:
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

    for block, layers_per_block in enumerate(args.layers):
        for _ in range(layers_per_block):
            x = densely_connected_block(x, args, args.dilation_rate[block])

        if block < len(args.layers) - 1:
            x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            if args.dilation_rate[block + 1] == 1:
                x = keras.layers.MaxPooling2D(2, strides=2)(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2D(
                round(x.shape.as_list()[-1] // args.reduction[block] / 32) * 32,
                kernel_size=1,
                kernel_initializer="he_normal",
                use_bias=False,
            )(x)

    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    if include_top:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.GlobalAvgPool2D()(x)
        x = keras.layers.Dense(
            num_classes, activation="softmax", kernel_initializer="he_normal"
        )(x)
    return keras.Model(inputs=input, outputs=x)


@registry.register_hparams(binary_densenet)
class binary_densenet28(HParams):
    epochs = 120
    batch_size = 256
    learning_rate = 0.004
    learning_factor = 0.1
    learning_steps = [100, 110]
    initial_filters = 64
    growth_rate = 64
    reduction = [2.7, 2.7, 2.2]
    dilation_rate = [1, 1, 1, 1]
    layers = [6, 6, 6, 5]
    quantizer = lq.quantizers.SteSign(clip_value=1.3)
    constraint = lq.constraints.WeightClip(clip_value=1.3)

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


@registry.register_hparams(binary_densenet)
class binary_densenet37(binary_densenet28):
    batch_size = 192
    learning_steps = [100, 110]
    reduction = [3.3, 3.3, 4]
    layers = [6, 8, 12, 6]


@registry.register_hparams(binary_densenet)
class binary_densenet37_dilated(binary_densenet37):
    epochs = 80
    batch_size = 256
    learning_steps = [60, 70]
    dilation_rate = [1, 1, 2, 4]


@registry.register_hparams(binary_densenet)
class binary_densenet45(binary_densenet28):
    epochs = 125
    batch_size = 384
    learning_rate = 0.008
    learning_steps = [80, 100]
    reduction = [2.7, 3.3, 4]
    layers = [6, 12, 14, 8]


def BinaryDenseNet28(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the Binary BinaryDenseNet 28 architecture.

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

    model = binary_densenet(
        binary_densenet28(),
        input_shape=input_shape,
        num_classes=classes,
        input_tensor=input_tensor,
        include_top=include_top,
    )

    # Load weights.
    if weights == "imagenet":
        # download appropriate file
        if include_top:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_28_weights.h5",
                file_hash="21fe3ca03eed244df9c41a2219876fcf03e73800932ec96a3e2a76af4747ac53",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_28_weights_notop.h5",
                file_hash="a376df1e41772c4427edd1856072b934a89bf293bf911438bf6f751a9b2a28f5",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model


def BinaryDenseNet37(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the Binary BinaryDenseNet 37 architecture.

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

    model = binary_densenet(
        binary_densenet37(),
        input_shape=input_shape,
        num_classes=classes,
        input_tensor=input_tensor,
        include_top=include_top,
    )

    # Load weights.
    if weights == "imagenet":
        # download appropriate file
        if include_top:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_37_weights.h5",
                file_hash="8056a5d52c3ed86a934893987d09a06f59a5166aa9bddcaedb050f111d0a7d76",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_37_weights_notop.h5",
                file_hash="4e12bca9fd27580a5b833241c4eb35d6cc332878c406048e6ca8dbbc78d59175",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model


def BinaryDenseNet37Dilated(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the Dilated BinaryDenseNet 37 architecture.

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

    model = binary_densenet(
        binary_densenet37_dilated(),
        input_shape=input_shape,
        num_classes=classes,
        input_tensor=input_tensor,
        include_top=include_top,
    )

    # Load weights.
    if weights == "imagenet":
        # download appropriate file
        if include_top:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_37_dilated_weights.h5",
                file_hash="15c1bcd79b8dc22971382fbf79acf364a3f51049d0e584a11533e6fdbb7363d3",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_37_dilated_weights_notop.h5",
                file_hash="eaf3eac19fc90708f56a27435fb06d0e8aef40e6e0411ff7a8eefbe479226e4f",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model


def BinaryDenseNet45(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    classes=1000,
):
    """Instantiates the Binary BinaryDenseNet 45 architecture.

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

    model = binary_densenet(
        binary_densenet45(),
        input_shape=input_shape,
        num_classes=classes,
        input_tensor=input_tensor,
        include_top=include_top,
    )

    # Load weights.
    if weights == "imagenet":
        # download appropriate file
        if include_top:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_45_weights.h5",
                file_hash="d00a0d26fbd2dba1bfba8c0306c770f3aeea5c370e99f963bb239bd916f72c37",
            )
        else:
            weights_path = utils.download_pretrained_model(
                model="binary_densenet",
                version="v0.1.0",
                file="binary_densenet_45_weights_notop.h5",
                file_hash="e72d5cc6b0afe4612f8be7b1f9bb48a53ba2c8468b57bf1266d2900c99fd2adf",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model
