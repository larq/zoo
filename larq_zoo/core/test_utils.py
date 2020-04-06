import tensorflow as tf

from larq_zoo.core import utils


def test_global_pool():
    def build_model(input_res=(32, 32), data_format="channels_first"):
        if data_format == "channels_first":
            input_shape = (3, input_res[0], input_res[1])
        else:
            input_shape = (input_res[0], input_res[1], 3)
        inp = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(6, 3, 2, data_format=data_format)(inp)
        x = utils.global_pool(x, data_format=data_format)
        x = tf.keras.layers.Dense(2)(x)
        return tf.keras.Model(inputs=inp, outputs=x)

    for data_format in ["channels_first", "channels_last"]:
        fixed_model = build_model()
        dynamic_model = build_model(input_res=(None, 32))

        for model in [fixed_model, dynamic_model]:
            output_shape = model.outputs[0].get_shape().as_list()
            assert output_shape == [None, 2]
