import larq as lq
import tensorflow as tf


class Bop(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, threshold=1e-5, gamma=1e-2, name="Bop", **kwargs):
        super().__init__(name=name, **kwargs)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        self._optimizer = optimizer
        self._set_hyper("threshold", threshold)
        self._set_hyper("gamma", gamma)

    def _create_slots(self, var_list):
        for var in var_list:
            if self.is_binary(var):
                self.add_slot(var, "m")

    def apply_gradients(self, grads_and_vars, name=None):
        bin_grads_and_vars = [(g, v) for g, v in grads_and_vars if self.is_binary(v)]
        fp_grads_and_vars = [(g, v) for g, v in grads_and_vars if not self.is_binary(v)]

        bin_train_op = super().apply_gradients(bin_grads_and_vars, name=name)
        fp_train_op = self._optimizer.apply_gradients(fp_grads_and_vars, name=name)

        return tf.group(bin_train_op, fp_train_op, name="train_with_bop")

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError()

    def _resource_apply_dense(self, grad, var):
        print(f"Applying Bop to {var}")
        var_dtype = var.dtype.base_dtype
        gamma = self._get_hyper("gamma", var_dtype)
        threshold = self._get_hyper("threshold", var_dtype)
        m = self.get_slot(var, "m")

        m_t = tf.assign(
            m, (1 - gamma) * m + gamma * grad, use_locking=self._use_locking
        )
        var_t = lq.quantizers.sign(-tf.sign(var * m_t - threshold) * var)
        return tf.assign(var, var_t, use_locking=self._use_locking).op

    @staticmethod
    def is_binary(var):
        return "/kernel" in var.name and "quant_" in var.name

    def get_config(self):
        config = {
            "threshold": self._serialize_hyperparameter("threshold"),
            "gamma": self._serialize_hyperparameter("gamma"),
        }
        return {**super().get_config(), **self._optimizer.get_config(), **config}
