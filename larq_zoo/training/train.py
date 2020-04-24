import functools
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Union

import click
import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import Field
from zookeeper.tf import Experiment

from larq_zoo.core import utils


class TrainLarqZooModel(Experiment):
    # Save model checkpoints.
    use_model_checkpointing: bool = Field(True)

    # Log metrics to Tensorboard.
    use_tensorboard: bool = Field(True)

    # Use a per-batch progress bar (as opposed to per-epoch).
    use_progress_bar: bool = Field(False)

    # How often to run validation.
    validation_frequency: int = Field(1)

    # Whether or not to save models at the end.
    save_weights: bool = Field(True)

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
        return Path(self.output_dir) / "model"

    metrics: List[Union[Callable[[tf.Tensor, tf.Tensor], float], str]] = Field(
        lambda: ["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]
    )

    loss = Field("sparse_categorical_crossentropy")

    @property
    def steps_per_epoch(self):
        return self.dataset.num_examples("train") // self.batch_size

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

        train_data, num_train_examples = self.dataset.train(
            decoders=self.preprocessing.decoders
        )
        train_data = (
            train_data.cache()
            .shuffle(10 * self.batch_size)
            .repeat()
            .map(
                functools.partial(self.preprocessing, training=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(self.batch_size)
            .prefetch(1)
        )

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
                self.model.load_weights(str(self.model_path))
                print(f"Loaded model from epoch {initial_epoch}.")

        click.secho(str(self))

        self.model.fit(
            train_data,
            epochs=self.epochs,
            steps_per_epoch=math.ceil(num_train_examples / self.batch_size),
            validation_data=validation_data,
            validation_steps=math.ceil(num_validation_examples / self.batch_size),
            validation_freq=self.validation_frequency,
            verbose=1 if self.use_progress_bar else 2,
            initial_epoch=initial_epoch,
            callbacks=self.callbacks,
        )

        # Save model, weights, and config JSON.
        if self.save_weights:
            self.model.save(str(Path(self.output_dir) / f"{self.model.name}.h5"))
            self.model.save_weights(
                str(Path(self.output_dir) / f"{self.model.name}_weights.h5")
            )
            with open(
                Path(self.output_dir) / f"{self.model.name}.json", "w"
            ) as json_file:
                json_file.write(self.model.to_json())
