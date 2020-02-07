import numpy as np
import pytest
import tensorflow as tf

from larq_zoo import preprocess_input


def test_numpy_input():
    image = np.random.randint(0, 255, size=(300, 300, 3), dtype="uint8")
    prepro = preprocess_input(image)
    assert isinstance(prepro, np.ndarray)


def test_tensor_input():
    image = np.random.randint(0, 255, size=(300, 300, 3), dtype="uint8")
    tf_image = tf.constant(image)
    prepro = preprocess_input(tf_image)
    assert isinstance(prepro, tf.Tensor)


def test_wrong_input():
    with pytest.raises(ValueError, match="Input must be of size .*"):
        preprocess_input(np.random.randint(0, 255, size=(4, 32, 32, 3), dtype="uint8"))
