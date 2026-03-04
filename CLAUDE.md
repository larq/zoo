# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Larq Zoo is a Python package providing reference implementations of Binarized Neural Networks (BNNs). It serves as both a library (for loading pre-trained models) and a CLI tool (for training models). Built on top of `larq` (BNN layers/quantizers), `tensorflow`, and `zookeeper` (experiment management/CLI).

## Common Commands

### Install for development
```bash
pip install -e ".[test]"
# Also need TensorFlow installed separately, e.g.:
pip install tensorflow-cpu==2.10.1
```

### Run tests
```bash
pytest . -n auto                              # all tests except training
pytest . -n auto --ignore=tests/train_test.py # same as CI
pytest tests/train_test.py -n auto            # training tests only
pytest tests/models_test.py -k "test_name"    # single test
```

### Lint and format
```bash
black . --check --target-version py310
isort . --check --diff
flake8
pytype --jobs auto
```

To auto-fix formatting:
```bash
black . --target-version py310
isort .
```

### Train a model (CLI)
```bash
lqz TrainQuickNet dataset=ImageNet epochs=600 batch_size=2048
```

## Architecture

### Package structure

- **`larq_zoo/core/`** - Base classes and utilities shared across all models
  - `model_factory.py`: `ModelFactory` base class that all model definitions extend (via zookeeper's `ComponentField`/`Field` system for declarative configuration)
  - `utils.py`: Weight downloading (from GitHub releases), input validation, ImageNet prediction decoding, custom global pooling optimized for Larq Compute Engine

- **`larq_zoo/literature/`** - Implementations of published BNN architectures (BinaryAlexNet, BiRealNet, BinaryDenseNet, DoReFaNet, MeliusNet, Real-to-Binary nets, BinaryResNetE18, XNORNet). Each file defines a `ModelFactory` subclass that builds the model and a public function (e.g. `BiRealNet(...)`) as the user-facing API.

- **`larq_zoo/sota/`** - Plumerai's state-of-the-art models (QuickNet, QuickNetSmall, QuickNetLarge) with pre-trained ImageNet weights.

- **`larq_zoo/training/`** - Training infrastructure
  - `train.py`: `TrainLarqZooModel` base experiment class
  - `data.py`: ImageNet-style preprocessing/augmentation
  - `datasets.py`: Dataset definitions (ImageNet, Cifar10, Mnist, OxfordFlowers)
  - `basic_experiments.py` / `sota_experiments.py`: Training configs for each model
  - `multi_stage_experiments.py`: Multi-stage training with knowledge distillation
  - `knowledge_distillation/`: Attention matching loss, multi-stage training phases

### Key patterns

- Models use the **zookeeper** framework for configuration: fields are declared with `Field()` and `ComponentField()`, enabling CLI-driven configuration and dependency injection.
- Pre-trained weights are hosted as **GitHub release assets** and downloaded via `tf.keras.utils.get_file` with SHA256 verification.
- The `lqz` CLI entry point (`larq_zoo/training/main.py`) uses zookeeper to discover and run experiment classes.

## Code Style

- Python 3.10+ target
- Formatting: **black** (target py310) + **isort** (profile: black)
- Linting: **flake8** + **pytype** for type checking
- Configuration in `setup.cfg`
