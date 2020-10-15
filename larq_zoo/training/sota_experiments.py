import larq as lq
import tensorflow as tf
from zookeeper import ComponentField, Field, cli, task

from larq_zoo.sota.quicknet import (
    QuickNetFactory,
    QuickNetLargeFactory,
    QuickNetSmallFactory,
)
from larq_zoo.training.learning_schedules import CosineDecayWithWarmup
from larq_zoo.training.train import TrainLarqZooModel


@task
class TrainQuickNet(TrainLarqZooModel):
    model = ComponentField(QuickNetFactory)
    epochs = Field(600)
    batch_size = Field(2048)

    @Field
    def optimizer(self):
        binary_opt = tf.keras.optimizers.Adam(
            learning_rate=CosineDecayWithWarmup(
                max_learning_rate=1e-2,
                warmup_steps=self.steps_per_epoch * 5,
                decay_steps=self.steps_per_epoch * self.epochs,
            )
        )
        fp_opt = tf.keras.optimizers.SGD(
            learning_rate=CosineDecayWithWarmup(
                max_learning_rate=0.1,
                warmup_steps=self.steps_per_epoch * 5,
                decay_steps=self.steps_per_epoch * self.epochs,
            ),
            momentum=0.9,
        )
        return lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, binary_opt),
            default_optimizer=fp_opt,
        )


@task
class TrainQuickNetSmall(TrainQuickNet):
    model = ComponentField(QuickNetSmallFactory)


@task
class TrainQuickNetLarge(TrainQuickNet):
    model = ComponentField(QuickNetLargeFactory)


if __name__ == "__main__":
    cli()
