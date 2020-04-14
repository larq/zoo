import functools
import math
from pathlib import Path

import click
import larq as lq
import tensorflow as tf

from larq_zoo.core import utils
from training.train import TrainLarqZooModel


class MultiStageLarqZooTraining(TrainLarqZooModel):



    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

            # if initial_epoch > 0:
            #     self.model.load_weights(self.model_path)
            #     print(f"Loaded model from epoch {initial_epoch}.")

        click.secho(str(self))

        self.model.fit(
            train_data,
            epochs=self.epochs,
            steps_per_epoch=math.ceil(num_train_examples / self.batch_size),
            validation_data=validation_data,
            validation_steps=math.ceil(num_validation_examples / self.batch_size),
            validation_freq=self.validation_frequency,
            verbose=1 if self.use_progress_bar else 2,
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