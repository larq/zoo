import pytest
import tensorflow as tf

from larq_zoo.core import utils


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
@pytest.mark.parametrize("input_res", [(32, 32), (None, 32)])
def test_global_pool(input_res, data_format):
    shape = (3, *input_res) if data_format == "channels_first" else (*input_res, 3)
    inp = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Conv2D(6, 3, 2, data_format=data_format)(inp)
    x = utils.global_pool(x, data_format=data_format)
    x = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inp, x)

    assert model.outputs[0].shape.as_list() == [None, 2]
