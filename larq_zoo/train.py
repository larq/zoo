import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Union

import click
import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import Field, cli
from zookeeper.tf import Experiment

from larq_zoo import utils


class TrainLarqZooModel(Experiment):
    # Save model checkpoints.
    use_model_checkpointing: bool = Field(True)

    # Log metrics to Tensorboard.
    use_tensorboard: bool = Field(True)

    # Use a per-batch progress bar (as opposed to per-epoch).
    use_progress_bar: bool = Field(False)

    # Where to store output.
    @Field
    def output_dir(self) -> Union[str, os.PathLike]:
        return (
            Path.home()
            / "zookeeper-logs"
            / self.dataset.__class__.__name__
            / self.__class__.__name__
            / datetime.now().strftime("%Y%m%d_%H%M")
        )

    @property
    def model_path(self):
        return self.output_dir / "model"

    learning_rate_schedule: Optional[Callable] = None

    metrics: List[Union[Callable[[tf.Tensor, tf.Tensor], float], str]] = Field(
        lambda: ["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]
    )

    loss = Field("sparse_categorical_crossentropy")

    @Field
    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        if self.use_model_checkpointing:
            callbacks.append(
                utils.ModelCheckpoint(
                    filepath=str(self.model_path), save_weights_only=True
                )
            )
        if hasattr(self, "learning_rate_schedule"):
            callbacks.append(
                keras.callbacks.LearningRateScheduler(self.learning_rate_schedule)
            )
        if self.use_tensorboard:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=self.output_dir, write_graph=False, profile_batch=0
                )
            )
        return callbacks

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        initial_epoch = utils.get_current_epoch(self.output_dir)

        validation_data, num_validation_examples = self.dataset.validation(
            decoders=self.preprocessing.decoders
        )
        validation_data = (
            validation_data.cache()
            .repeat()
            .map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(1)
        )

        with utils.get_distribution_scope(self.batch_size):
            self.model.compile(
                optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
            )

            lq.models.summary(self.model)

            if initial_epoch > 0:
                self.model.load_weights(self.model_path)
                print(f"Loaded model from epoch {initial_epoch}.")

        click.secho(str(self))

        self.model.evaluate(
            validation_data,
            steps=math.ceil(num_validation_examples / self.batch_size),
            verbose=1 if self.use_progress_bar else 2,
            callbacks=self.callbacks,
        )

        # Evaluate loss metrics
        metric_dict = {}
        for metric, name in zip(self.model.metrics, self.model.metrics_names[1:]):
            value = metric.result().numpy()
            metric_dict[f"val_{name}"] = value
            tf.summary.scalar(f"epoch_{name}", value, step=0)
        print(" - ".join(f"{name}: {value}" for name, value in metric_dict.items()))


if __name__ == "__main__":
    import importlib

    # Running it without the CLI requires us to first import larq_zoo
    # in order to register the models and datasets
    importlib.import_module("larq_zoo")
    cli()
