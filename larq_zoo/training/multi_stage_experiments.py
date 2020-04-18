from typing import Sequence

import tensorflow as tf
from zookeeper import ComponentField, Field, cli, task

from larq_zoo.literature.real_to_bin_nets import (
    RealToBinNetBANFactory,
    RealToBinNetBNNFactory,
    RealToBinNetFPFactory,
    ResNet18FPFactory,
    StrongBaselineNetBAN,
    StrongBaselineNetBNN,
)
from larq_zoo.training.datasets import ImageNet
from larq_zoo.training.knowledge_distillation.multi_stage_training import (
    LarqZooModelTrainingPhase,
    MultiStageExperiment,
)

# --------- Learning rate schedules -------------


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


# --------- Real-to-Binary: Strong Baseline Model training -------------


@task
class TrainR2BStrongBaselineBAN(LarqZooModelTrainingPhase):
    stage = Field(0)

    dataset = ComponentField(ImageNet)

    learning_rate: float = Field(1e-3)
    learning_rate_decay: float = Field(0.1)
    epochs: int = Field(75)
    batch_size: int = Field(8)
    amount_of_images: int = Field(1281167)
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


# --------- Real-to-Binary: Full Model training -------------


@task
class TrainFPResnet18(LarqZooModelTrainingPhase):
    stage = Field(0)
    dataset = ComponentField(ImageNet)
    learning_rate: float = Field(1e-1)
    epochs: int = Field(100)
    batch_size: int = Field(512)
    amount_of_images: int = Field(1281167)
    warmup_duration: int = Field(5)

    @property
    def steps_per_epoch(self):
        return self.amount_of_images // self.batch_size

    optimizer = Field(
        lambda self: tf.keras.optimizers.SGD(
            CosineDecayWithWarmup(
                max_learning_rate=self.learning_rate,
                warmup_steps=self.warmup_duration * self.steps_per_epoch,
                decay_steps=(self.epochs - self.warmup_duration) * self.steps_per_epoch,
            )
        )
    )

    student_model = ComponentField(ResNet18FPFactory)


@task
class TrainR2BBFP(TrainFPResnet18):
    stage = Field(1)
    learning_rate: float = Field(1e-3)
    learning_rate_decay: float = Field(0.3)  # TODO double check
    epochs: int = Field(75)
    batch_size: int = Field(256)

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            R2BStepSchedule(
                initial_learning_rate=self.learning_rate,
                steps_per_epoch=self.steps_per_epoch,
                decay_fraction=self.learning_rate_decay,
            )
        )
    )

    teacher_model = ComponentField(ResNet18FPFactory)
    initialize_teacher_weights_from = Field("resnet_fp")
    student_model = ComponentField(RealToBinNetFPFactory)

    classification_weight = Field(1.0)  # TODO double check
    attention_matching_weight = Field(30.0)  # TODO double check
    output_matching_weight = Field(3.0)  # TODO double check

    attention_matching_volume_names = Field(
        lambda: [f"block_{b}_out" for b in range(2, 10)]
    )


@task
class TrainR2BBAN(TrainR2BBFP):
    stage = Field(2)
    learning_rate: float = Field(1e-3)

    classification_weight = Field(1.0)  # TODO double check
    attention_matching_weight = Field(30.0)  # TODO double check
    output_matching_weight = Field(3.0)  # TODO double check
    attention_matching_train_teacher = Field(False)
    output_matching_train_teacher = Field(False)

    teacher_model = ComponentField(RealToBinNetFPFactory)
    student_model = ComponentField(RealToBinNetBANFactory)

    initialize_teacher_weights_from = Field("r2b_fp")


@task
class TrainR2BBNN(TrainR2BBFP):
    stage = Field(3)
    learning_rate: float = Field(2e-4)

    classification_weight = Field(1.0)  # TODO double check
    attention_matching_weight = Field(0.0)  # TODO double check
    output_matching_weight = Field(1.25)  # TODO double check
    output_matching_softmax_temperature = Field(3.0)  # TODO double check
    attention_matching_train_teacher = Field(False)  # TODO default?
    output_matching_train_teacher = Field(False)

    teacher_model = ComponentField(RealToBinNetBANFactory)
    student_model = ComponentField(RealToBinNetBNNFactory)

    initialize_teacher_weights_from = Field("r2b_ban")
    initialize_student_weights_from = Field("r2b_ban")


@task
class TrainR2BBNNAlternative(TrainR2BBNN):
    """We deviate slightly from Martinez et. al. here"""

    epochs = Field(100)
    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            CosineDecayWithWarmup(
                max_learning_rate=1.75e-4,
                warmup_steps=self.steps_per_epoch * 10,
                decay_steps=self.steps_per_epoch * (self.epochs - 10),
            )
        )
    )


@task
class TrainR2B(MultiStageExperiment):
    stage_0 = ComponentField(TrainFPResnet18)
    stage_1 = ComponentField(TrainR2BBFP)
    stage_2 = ComponentField(TrainR2BBAN)
    stage_3 = ComponentField(TrainR2BBNNAlternative)


if __name__ == "__main__":
    cli()
