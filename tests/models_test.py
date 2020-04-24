import functools
import os

import numpy as np
import pytest
from tensorflow import keras

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
            (lqz.literature.MeliusNet22, 512),
            (lqz.literature.XNORNet, 4096),
            (lqz.literature.DoReFaNet, 256),
            (lqz.literature.RealToBinaryNet, 512),
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

    notop_model = app(weights="imagenet", include_top=False)
    for weight, notop_weight in zip(model.get_weights(), notop_model.get_weights()):
        np.testing.assert_allclose(notop_weight, weight)


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
