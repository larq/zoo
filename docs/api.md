# Larq Zoo Pretrained Models

Larq Zoo provides reference implementations of deep neural networks with extremely low precision weights and activations that are made available alongside pre-trained weights.
These models can be used for prediction, feature extraction, and fine-tuning.

Weights can be downloaded automatically when instantiating a model. They are stored at `~/.larq/models/`.

## Available models for image classification with weights trained on ImageNet

The Top-1 and Top-5 accuracy refers to the model's performance on the [ImageNet](http://image-net.org/) validation dataset.

| Model                     | Top-1 Accuracy | Top-5 Accuracy |
| ------------------------- | -------------- | -------------- |
| [Bi-Real Net](#BiRealNet) | ? %            | ? %            |

---

## Usage examples for image classification models

### Classify ImageNet classes with Bi-Real Net

```python
import tensorflow as tf
import larq_zoo as lqz

model = lqz.BiRealNet(weights="imagenet")

img_path = "tests/fixtures/elephant.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print("Predicted:", lqz.decode_predictions(preds, top=3)[0])
# Predicted: [("n01871265", "tusker", 0.7427464), ("n02504458", "African_elephant", 0.19439144), ("n02504013", "Indian_elephant", 0.058899447)]
```

### Extract features with Bi-Real Net

```python
import tensorflow as tf
import larq_zoo as lqz

model = lqz.BiRealNet(weights="imagenet", include_top=False)

img_path = "tests/fixtures/elephant.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

features = model.predict(x)
```

### Extract features from an arbitrary intermediate layer with Bi-Real Net

```python
import tensorflow as tf
import larq_zoo as lqz

base_model = lqz.BiRealNet(weights="imagenet")
model = tf.keras.models.Model(
    inputs=base_model.input, outputs=base_model.get_layer("average_pooling2d_8").output
)

img_path = "tests/fixtures/elephant.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

average_pool_8_features = model.predict(x)
```

### Fine-tune Bi-Real Net on a new set of classes

```python
import tensorflow as tf
import larq as lq
import larq_zoo as lqz

# create the base pre-trained model
base_model = lqz.BiRealNet(weights="imagenet", include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# let's add a binarized fully-connected layer
x = lq.layers.QuantDense(
    1024,
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip",
    use_bias=False,
    activation="relu",
)(x)
x = tf.keras.layers.BatchNormalization()(x)
# and a full precision logistic layer -- let's say we have 200 classes
predictions = tf.keras.layers.Dense(200, activation="softmax")(x)

# this is the model we will train
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Bi-Real Net layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

# train the model on the new data for a few epochs
model.fit(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from Bi-Real Net. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top block, i.e. we will freeze
# the first 49 layers and unfreeze the rest:
for layer in model.layers[:49]:
   layer.trainable = False
for layer in model.layers[49:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
    loss="categorical_crossentropy",
)

# we train our model again (this time fine-tuning the top block
# alongside the top Dense layers
model.fit(...)
```

### Build Bi-Real Net over a custom input Tensor

```python
import tensorflow as tf
import larq_zoo as lqz

# this could also be the output a different Keras model or layer
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))

model = lqz.BiRealNet(input_tensor=input_tensor, weights="imagenet")
```

## Larq Zoo API Documentation

### BiRealNet<a class="headerlink code-link" style="float:right;" href="https://github.com/plumerai/larq-zoo/blob/master/larq_zoo/birealnet.py#L99" title="Source code"></a>

```python
BiRealNet(include_top=True,
          weights="imagenet",
          input_tensor=None,
          input_shape=None,
          classes=1000)
```

Instantiates the Bi-Real Net architecture.

Optionally loads weights pre-trained on ImageNet.

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

### decode_predictions<a class="headerlink code-link" style="float:right;" href="https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L198" title="Source code"></a>

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

### preprocess_input<a class="headerlink code-link" style="float:right;" href="https://github.com/plumerai/larq-zoo/blob/master/larq_zoo/data.py#L33" title="Source code"></a>

```python
preprocess_input(image)
```

Preprocesses a Tensor or Numpy array encoding a image.

**Arguments**

- `image`: Numpy array or symbolic Tensor, 3D.

**Returns**

Preprocessed Tensor or Numpy array.
