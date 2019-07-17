# Larq Zoo Pretrained Models

Larq Zoo provides reference implementations of deep neural networks with extremely low precision weights and activations that are made available alongside pre-trained weights.
These models can be used for prediction, feature extraction, and fine-tuning.

We believe that a collection of tested implementations with pretrained weights is greatly beneficial for the field of Extremely Quantized Neural Networks. To improve reproducibility we have implemented a few commonly used models found in the literature. If you have developed or reimplemented a Binarized or other Extremely Quantized Neural Network and want to share it with the community such that future papers can build on top of your work, please add it to Larq Zoo or get in touch with us if you need any help.

## Installation

Larq Zoo is not included in Larq by default. To start using it, you can install it with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq-zoo
```

Weights can be downloaded automatically when instantiating a model. They are stored at `~/.larq/models/`.

## Available models

The following models are trained on the [ImageNet](http://image-net.org/) dataset. The Top-1 and Top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

The model definitions and the train loop are available in the [Larq Zoo repository](https://github.com/larq/zoo).

| Model                                        | Top-1 Accuracy | Top-5 Accuracy |
| -------------------------------------------- | -------------- | -------------- |
| [Binary AlexNet](/models/api/#binaryalexnet) | 36.28 %        | 61.05 %        |
| [Bi-Real Net](/models/api/#birealnet)        | 55.88 %        | 78.62 %        |
| [XNOR-Net](/models/api/#xnornet)             | 43.03 %        | 67.32 %        |
