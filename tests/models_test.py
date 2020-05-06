from pathlib import Path

import larq as lq
import numpy as np
import pytest
import tensorflow as tf
from zookeeper import cli

import larq_zoo as lqz


@pytest.fixture(autouse=True)
def run_around_tests():
    tf.keras.backend.clear_session()
    yield


parametrize = pytest.mark.parametrize(
    "app,last_feature_dim",
    [
        (lqz.literature.BinaryAlexNet, 256),
        (lqz.literature.BiRealNet, 512),
        (lqz.literature.BinaryResNetE18, 512),
        (lqz.literature.BinaryDenseNet28, 576),
        (lqz.literature.BinaryDenseNet37, 640),
        (lqz.literature.BinaryDenseNet37Dilated, 640),
        (lqz.literature.BinaryDenseNet45, 800),
        (lqz.literature.MeliusNet22, 512),
        (lqz.literature.XNORNet, 4096),
        (lqz.literature.DoReFaNet, 256),
        (lqz.literature.RealToBinaryNet, 512),
        (lqz.sota.QuickNet, 512),
        (lqz.sota.QuickNetLarge, 512),
        (lqz.sota.QuickNetXL, 512),
    ],
)


@parametrize
def test_prediction(app, last_feature_dim):
    img = tf.keras.preprocessing.image.load_img(
        Path() / "tests" / "fixtures" / "elephant.jpg"
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = lqz.preprocess_input(img)
    model = app(weights="imagenet")
    preds = model.predict(np.expand_dims(img, axis=0))

    # Test correct label is in top 3 (weak correctness test).
    names = [p[1] for p in lqz.decode_predictions(preds, top=3)[0]]
    assert "African_elephant" in names

    notop_model = app(weights="imagenet", include_top=False)
    for weight, notop_weight in zip(model.get_weights(), notop_model.get_weights()):
        np.testing.assert_allclose(notop_weight, weight)


@parametrize
def test_basic(app, last_feature_dim):
    model = app(weights=None)
    assert model.output_shape == (None, 1000)


@parametrize
def test_keras_tensor_input(app, last_feature_dim):
    input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
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
def test_model_summary(app, last_feature_dim):
    input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
    model = app(weights=None, input_tensor=input_tensor)

    class PrintToVariable:
        output = ""

        def __call__(self, x):
            self.output += f"{x}\n"

    capture = PrintToVariable()
    lq.models.summary(model, print_fn=capture)

    summary_file = (
        Path()
        / "tests"
        / "snapshots"
        / "model_summaries"
        / f"{app.__name__}_{last_feature_dim}.txt"
    )

    if summary_file.exists():
        with open(summary_file, "r") as file:
            content = file.read()
        assert content.lower() == capture.output.lower()
    else:
        with open(summary_file, "w") as file:
            file.write(capture.output)
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
