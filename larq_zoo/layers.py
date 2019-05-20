import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import nn_ops
import larq as lq


@lq.utils.register_keras_custom_object
class LatentFree(tf.keras.constraints.Constraint):
    """Latent free constraint"""

    def __init__(self, threshold=1e-7):
        self.threshold = threshold

    def __call__(self, x):
        return tf.sign(tf.abs(x) - (1 - self.threshold)) * tf.sign(x)

    def get_config(self):
        return {"threshold": self.threshold}


def flipout(x, rate=0.2, noise_shape=None, seed=None):
    @tf.custom_gradient
    def _flipout(x):
        keep_prob = 1 - rate
        random_tensor = keep_prob + tf.random.uniform(
            noise_shape, seed=seed, dtype=x.dtype
        )
        binary_tensor = tf.math.floor(random_tensor)
        flip_tensor = binary_tensor * 2 - 1

        def grad(dy):
            return binary_tensor * dy

        return flip_tensor * x, grad

    return _flipout(x)


@lq.utils.register_keras_custom_object
class Flipout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def flipped_inputs():
            return flipout(
                inputs,
                rate=self.rate,
                noise_shape=nn_ops._get_noise_shape(inputs, self.noise_shape),
                seed=self.seed,
            )

        output = tf_utils.smart_cond(
            training, flipped_inputs, lambda: tf.identity(inputs)
        )
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"rate": self.rate, "noise_shape": self.noise_shape, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
