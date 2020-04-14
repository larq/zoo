from typing import Sequence

import tensorflow as tf

from larq_zoo.literature.real_to_bin_nets import (
    StrongBaselineNetBAN,
    StrongBaselineNetBNN,
)
from larq_zoo.training.datasets import ImageNet
from zookeeper import ComponentField, Field, cli, task

from larq_zoo.training.knowledge_distillation.multi_stage_training import (
    LarqZooModelTrainingPhase,
    MultiStageExperiment,
)


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


@task
class TrainR2BStrongBaselineBAN(LarqZooModelTrainingPhase):
    stage = Field(0)

    dataset = ComponentField(ImageNet)

    learning_rate: float = Field(1e-3)
    learning_rate_decay: float = Field(0.1)
    # epochs: int = Field(75)
    # batch_size: int = Field(256) # TODO
    epochs: int = Field(6)
    batch_size: int = Field(8)
    amount_of_images: int = Field(256)
    # amount_of_images: int = Field(1281167) #TODO
    warmup_duration: int = Field(5)

    @property
    def steps_per_epoch(self):
        return self.amount_of_images // self.batch_size

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            R2BStepSchedule(
                initial_learning_rate=self.learning_rate,
                steps_per_epoch=self.steps_per_epoch,
                decay_fraction=self.learning_rate_decay,
            )
        )
    )

    student_model = ComponentField(StrongBaselineNetBAN)


@task
class TrainR2BStrongBaselineBNN(TrainR2BStrongBaselineBAN):
    stage = Field(1)
    learning_rate: float = Field(2e-4)
    student_model = ComponentField(StrongBaselineNetBNN)
    initialize_student_weights_from = Field("baseline_ban")


@task
class TrainR2BStrongBaseline(MultiStageExperiment):
    stage_0 = ComponentField(TrainR2BStrongBaselineBAN)
    stage_1 = ComponentField(TrainR2BStrongBaselineBNN)


if __name__ == "__main__":
    cli()
