import functools
import os
from pathlib import Path

import larq as lq
import numpy as np
import pytest
from tensorflow import keras
from zookeeper import cli

import larq_zoo as lqz


def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        keras.backend.clear_session()
        return output

    return wrapper


def parametrize(func):
    func = keras_test(func)
    return pytest.mark.parametrize(
        "app,last_feature_dim",
        [
            (lqz.literature.BinaryAlexNet, 256),
            (lqz.literature.BiRealNet, 512),
            (lqz.literature.BinaryResNetE18, 512),
            (lqz.literature.BinaryDenseNet28, 576),
            (lqz.literature.BinaryDenseNet37, 640),
            (lqz.literature.BinaryDenseNet37Dilated, 640),
            (lqz.literature.BinaryDenseNet45, 800),
            (lqz.literature.XNORNet, 4096),
            (lqz.literature.DoReFaNet, 256),
            (lqz.sota.QuickNet, 512),
            (lqz.sota.QuickNetLarge, 512),
            (lqz.sota.QuickNetXL, 512),
        ],
    )(func)


@parametrize
def test_prediction(app, last_feature_dim):
    file = os.path.join(os.path.dirname(__file__), "fixtures", "elephant.jpg")
    img = keras.preprocessing.image.load_img(file)
    img = keras.preprocessing.image.img_to_array(img)
    img = lqz.preprocess_input(img)
    model = app(weights="imagenet")
    preds = model.predict(np.expand_dims(img, axis=0))

    # Test correct label is in top 3 (weak correctness test).
    names = [p[1] for p in lqz.decode_predictions(preds, top=3)[0]]
    assert "African_elephant" in names


@parametrize
def test_basic(app, last_feature_dim):
    model = app(weights=None)
    assert model.output_shape == (None, 1000)


@parametrize
def test_keras_tensor_input(app, last_feature_dim):
    input_tensor = keras.layers.Input(shape=(224, 224, 3))
    model = app(weights=None, input_tensor=input_tensor)
    assert model.output_shape == (None, 1000)


@parametrize
def test_no_top(app, last_feature_dim):
    model = app(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, last_feature_dim)


@parametrize
def test_no_top_variable_shape_1(app, last_feature_dim):
    input_shape = (None, None, 1)
    model = app(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, last_feature_dim)


@parametrize
def test_no_top_variable_shape_4(app, last_feature_dim):
    input_shape = (None, None, 4)
    model = app(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, last_feature_dim)


@parametrize
def test_model_summary(app, last_feature_dim, capsys, snapshot):
    input_tensor = keras.layers.Input(shape=(224, 224, 3))
    model = app(weights=None, input_tensor=input_tensor)
    lq.models.summary(model)
    out, err = capsys.readouterr()

    summary_file = (
        Path(__file__).parent
        / "snapshots"
        / "model_summaries"
        / f"{app.__name__}_{last_feature_dim}.txt"
    )

    if summary_file.exists():
        with open(summary_file, "r") as file:
            content = file.read()
        assert content == out
    else:
        with open(summary_file, "w") as file:
            file.write(out)
        raise FileNotFoundError(
            f"Could not find snapshot {summary_file}, so generated a new summary. "
            "If this was intentional, re-running the tests locally will make them pass."
        )


@pytest.mark.parametrize("command_name", cli.commands.keys())
def test_experiments(command_name: str, snapshot, capsys):
    try:
        cli.commands[command_name](
            ["dataset=DummyOxfordFlowers", "batch_size=2", "dry_run=True"]
        )
    # Catch successful SystemExit to prevent exception
    except SystemExit as e:
        if e.code != 0:
            raise
