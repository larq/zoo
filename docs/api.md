# Larq Zoo API Documentation

## BinaryAlexNet<a class="headerlink code-link" style="float:right;" href="https://github.com/larq/zoo/blob/master/larq_zoo/binarynet.py#L88" title="Source code"></a>

```python
BinaryAlexNet(include_top=True,
              weights="imagenet",
              input_tensor=None,
              input_shape=None,
              classes=1000)
```

Instantiates the Binary AlexNet architecture ([interactive architecture diagram](https://lutzroeder.github.io/netron/?url=https://cors-anywhere.herokuapp.com/https://github.com/larq/zoo/releases/download/binary_alexnet-v0.2.0/binary_alexnet.h5)).

Optionally loads weights pre-trained on ImageNet.

```plot-altair
/plots/binary_alexnet.vg.json
```

**Arguments**

- `include_top`: whether to include the fully-connected layers at the top of the network.
- `weights`: one of `None` (random initialization), `"imagenet"` (pre-training on
  ImageNet), or the path to the weights file to be loaded.
- `input_tensor`: optional Keras Tensor (i.e. output of `layers.Input()`) to use as
  image input for the model.
- `input_shape`: optional shape tuple, only to be specified if `include_top` is False
  (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data
  format) or `(3, 224, 224)` (with `channels_first` data format).
  It should have exactly 3 inputs channels.
- `classes`: optional number of classes to classify images into, only to be specified
  if `include_top` is True, and if no `weights` argument is specified.

**Returns**

A Keras model instance.

**Raises**

- **ValueError**: in case of invalid argument for `weights`, or invalid input shape.

**References**

- [Binarized Neural Networks](https://papers.nips.cc/paper/6573-binarized-neural-networks)

## BiRealNet<a class="headerlink code-link" style="float:right;" href="https://github.com/larq/zoo/blob/master/larq_zoo/birealnet.py#L100" title="Source code"></a>

```python
BiRealNet(include_top=True,
          weights="imagenet",
          input_tensor=None,
          input_shape=None,
          classes=1000)
```

Instantiates the Bi-Real Net architecture ([interactive architecture diagram](https://lutzroeder.github.io/netron/?url=https://cors-anywhere.herokuapp.com/https://github.com/larq/zoo/releases/download/birealnet-v0.2.0/birealnet.h5)).

Optionally loads weights pre-trained on ImageNet.

```plot-altair
/plots/birealnet.vg.json
```

**Arguments**

- `include_top`: whether to include the fully-connected layer at the top of the network.
- `weights`: one of `None` (random initialization), `"imagenet"` (pre-training on
  ImageNet), or the path to the weights file to be loaded.
- `input_tensor`: optional Keras Tensor (i.e. output of `layers.Input()`) to use as
  image input for the model.
- `input_shape`: optional shape tuple, only to be specified if `include_top` is False
  (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data
  format) or `(3, 224, 224)` (with `channels_first` data format).
  It should have exactly 3 inputs channels.
- `classes`: optional number of classes to classify images into, only to be specified
  if `include_top` is True, and if no `weights` argument is specified.

**Returns**

A Keras model instance.

**Raises**

- **ValueError**: in case of invalid argument for `weights`, or invalid input shape.

**References**

- [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
  Representational Capability and Advanced Training
  Algorithm](https://arxiv.org/abs/1808.00278)

## XNORNet<a class="headerlink code-link" style="float:right;" href="https://github.com/larq/zoo/blob/master/larq_zoo/xnornet.py#L150" title="Source code"></a>

```python
XNORNet(include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        classes=1000)
```

Instantiates the XNOR-Net architecture ([interactive architecture diagram](https://lutzroeder.github.io/netron/?url=https://cors-anywhere.herokuapp.com/https://github.com/larq/zoo/releases/download/xnornet-v0.2.0/xnornet.h5)).

Optionally loads weights pre-trained on ImageNet.

```plot-altair
/plots/xnornet.vg.json
```

**Arguments**

- `include_top`: whether to include the fully-connected layer at the top of the network.
- `weights`: one of `None` (random initialization), `"imagenet"` (pre-training on
  ImageNet), or the path to the weights file to be loaded.
- `input_tensor`: optional Keras Tensor (i.e. output of `layers.Input()`) to use as
  image input for the model.
- `input_shape`: optional shape tuple, only to be specified if `include_top` is False
  (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data
  format) or `(3, 224, 224)` (with `channels_first` data format).
  It should have exactly 3 inputs channels.
- `classes`: optional number of classes to classify images into, only to be specified
  if `include_top` is True, and if no `weights` argument is specified.

**Returns**

A Keras model instance.

**Raises**

- **ValueError**: in case of invalid argument for `weights`, or invalid input shape.

**References**

- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)

## decode_predictions<a class="headerlink code-link" style="float:right;" href="https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L198" title="Source code"></a>

```python
decode_predictions(preds, top=5, **kwargs)
```

Decodes the prediction of an ImageNet model.

**Arguments**

- `preds`: Numpy Tensor encoding a batch of predictions.
- `top`: Integer, how many top-guesses to return.

**Returns**

A list of lists of top class prediction tuples `(class_name, class_description, score)`. One list of tuples per sample in batch input.

**Raises**

- **ValueError**: In case of invalid shape of the `pred` array (must be 2D).

## preprocess_input<a class="headerlink code-link" style="float:right;" href="https://github.com/larq/zoo/blob/master/larq_zoo/data.py#L33" title="Source code"></a>

```python
preprocess_input(image)
```

Preprocesses a Tensor or Numpy array encoding a image.

**Arguments**

- `image`: Numpy array or symbolic Tensor, 3D.

**Returns**

Preprocessed Tensor or Numpy array.
