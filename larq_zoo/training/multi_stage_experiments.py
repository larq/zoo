import tensorflow as tf
from zookeeper import ComponentField, Field, cli, task

from larq_zoo.literature.real_to_bin_nets import (
    RealToBinNetBANFactory,
    RealToBinNetBNNFactory,
    RealToBinNetFPFactory,
    ResNet18FPFactory,
    StrongBaselineNetBANFactory,
    StrongBaselineNetBNNFactory,
)
from larq_zoo.training.datasets import ImageNet
from larq_zoo.training.knowledge_distillation.multi_stage_training import (
    LarqZooModelTrainingPhase,
    MultiStageExperiment,
)
from larq_zoo.training.learning_schedules import CosineDecayWithWarmup, R2BStepSchedule

# --------- Real-to-Binary: Strong Baseline Model training -------------


@task
class TrainR2BStrongBaselineBAN(LarqZooModelTrainingPhase):
    stage = Field(0)

    dataset = ComponentField(ImageNet)

    learning_rate: float = Field(1e-3)
    learning_rate_decay: float = Field(0.1)
    epochs: int = Field(75)
    batch_size: int = Field(8)
    # amount_of_images: int = Field(1281167)
    warmup_duration: int = Field(5)

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            R2BStepSchedule(
                initial_learning_rate=self.learning_rate,
                steps_per_epoch=self.steps_per_epoch,
                decay_fraction=self.learning_rate_decay,
            )
        )
    )

    student_model = ComponentField(StrongBaselineNetBANFactory)


@task
class TrainR2BStrongBaselineBNN(TrainR2BStrongBaselineBAN):
    stage = Field(1)
    learning_rate: float = Field(2e-4)
    student_model = ComponentField(StrongBaselineNetBNNFactory)
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
    # amount_of_images: int = Field(1281167)
    warmup_duration: int = Field(5)

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
    learning_rate_decay: float = Field(0.3)
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

    classification_weight = Field(1.0)
    attention_matching_weight = Field(30.0)
    output_matching_weight = Field(3.0)

    attention_matching_volume_names = Field(
        lambda: [f"block_{b}_out" for b in range(2, 10)]
    )


@task
class TrainR2BBAN(TrainR2BBFP):
    stage = Field(2)
    learning_rate: float = Field(1e-3)

    teacher_model = ComponentField(RealToBinNetFPFactory)
    student_model = ComponentField(RealToBinNetBANFactory)

    initialize_teacher_weights_from = Field("r2b_fp")


@task
class TrainR2BBNN(TrainR2BBFP):
    stage = Field(3)
    learning_rate: float = Field(2e-4)

    classification_weight = Field(1.0)
    attention_matching_weight = Field(0.0)
    output_matching_weight = Field(0.8)
    output_matching_softmax_temperature = Field(1.0)

    teacher_model = ComponentField(RealToBinNetBANFactory)
    student_model = ComponentField(RealToBinNetBNNFactory)

    initialize_teacher_weights_from = Field("r2b_ban")
    initialize_student_weights_from = Field("r2b_ban")


@task
class TrainR2BBNNAlternative(TrainR2BBNN):
    """We deviate slightly from Martinez et. al. here"""

    warmup_duration = Field(10)
    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            CosineDecayWithWarmup(
                max_learning_rate=self.learning_rate,
                warmup_steps=self.steps_per_epoch * self.warmup_duration,
                decay_steps=self.steps_per_epoch * (self.epochs - self.warmup_duration),
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
