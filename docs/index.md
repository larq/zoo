# Larq Zoo Pretrained Models

Larq Zoo provides reference implementations of deep neural networks with extremely low precision weights and activations that are made available alongside pre-trained weights.
These models can be used for prediction, feature extraction, and fine-tuning.

## Installation

Larq Zoo is not included in Larq by default. To start using is, you can install it with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq-zoo
```

Weights can be downloaded automatically when instantiating a model. They are stored at `~/.larq/models/`.

## Available models

The following models are trained on the [ImageNet](http://image-net.org/) dataset. The Top-1 and Top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

The model definitions and the train loop are available in the [Larq Zoo repository](https://github.com/plumerai/larq-zoo).

| Model                                             | Top-1 Accuracy | Top-5 Accuracy |
| ------------------------------------------------- | -------------- | -------------- |
| [Bi-Real Net](https://larq.dev/models/#birealnet) | 55.88 %        | 78.62 %        |
