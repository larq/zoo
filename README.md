# Larq Zoo

[![GitHub Actions](https://github.com/larq/zoo/workflows/Unittest/badge.svg)](https://github.com/larq/zoo/actions?workflow=Unittest) [![Codecov](https://img.shields.io/codecov/c/github/larq/zoo)](https://codecov.io/github/larq/zoo?branch=master) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/larq-zoo.svg)](https://pypi.org/project/larq-zoo/) [![PyPI](https://img.shields.io/pypi/v/larq-zoo.svg)](https://pypi.org/project/larq-zoo/) [![Join the community on Spectrum](https://withspectrum.github.io/badge/badge.svg)](https://spectrum.chat/larq) [![PyPI - License](https://img.shields.io/pypi/l/larq-zoo.svg)](https://github.com/plumerai/larq-zoo/blob/master/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Larq Zoo consists of a `literature` and a `sota` submodule.

The `literature` submodule contains replications from research papers (all current models).
These models are intended to provide a stable reference for ideas presented in specific papers.
The model implementations will be maintained, but we will not attempt to improve these models over time by applying new training strategies or architecture innovations.

The `sota` submodule contains top models for various scenarios. These models are intended to use in a [`SW 2.0`](https://medium.com/@karpathy/software-2-0-a64152b37c35)-like fashion.
We will do our best to continuously improve the models, meaning their weights and even details about there architecture may change from release to release.

## Requirements

Before installing Larq Zoo, please install:

- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `1.14+` or `2.0.0`

## Installation

You can install Larq Zoo with Python's [pip](https://pip.pypa.io/en/stable/) package manager:

```shell
pip install larq-zoo
```
