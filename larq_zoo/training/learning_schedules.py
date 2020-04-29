import tensorflow as tf


class R2BStepSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, initial_learning_rate, steps_per_epoch, decay_fraction, name=None
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.decay_fraction = decay_fraction
        self.name = name

        self.warm_up_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            0.0,
            5 * steps_per_epoch,
            end_learning_rate=initial_learning_rate,
            power=1.0,
        )
        self.decay_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[
                40 * self.steps_per_epoch,
                60 * self.steps_per_epoch,
                70 * self.steps_per_epoch,
            ],
            values=[
                self.initial_learning_rate * self.decay_fraction ** i for i in range(4)
            ],
        )

    def __call__(self, step):
        return tf.cond(
            step <= 5 * self.steps_per_epoch,
            lambda: self.warm_up_schedule(step),
            lambda: self.decay_schedule(step),
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "steps_per_epoch": self.steps_per_epoch,
            "decay_fraction": self.decay_fraction,
            "name": self.name,
        }


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_learning_rate, warmup_steps, decay_steps):
        super().__init__()
        self.max_learning_rate = max_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warm_up_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=warmup_steps,
            end_learning_rate=max_learning_rate,
            power=1.0,
        )
        self.decay_schedule = tf.keras.experimental.CosineDecay(
            initial_learning_rate=max_learning_rate, decay_steps=decay_steps
        )

    def __call__(self, step):
        return tf.cond(
            step <= self.warmup_steps,
            lambda: self.warm_up_schedule(step),
            lambda: self.decay_schedule(step - self.warmup_steps),
        )

    def get_config(self):
        return {
            "max_learning_rate": self.max_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
        }
