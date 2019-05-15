from larq_flock import cli, build_train
from os import path
import click


@cli.command()
@click.option("--tensorboard/--no-tensorboard", default=True)
@build_train
def train(build_model, dataset, hparams, output_dir, epochs, tensorboard):
    import larq as lq
    from larq_zoo import utils
    import tensorflow as tf

    initial_epoch = utils.get_current_epoch(output_dir)
    model_path = path.join(output_dir, "model.h5")
    callbacks = [utils.ModelCheckpoint(filepath=model_path)]
    if hasattr(hparams, "learning_rate_schedule"):
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(hparams.learning_rate_schedule)
        )
    if tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=output_dir, write_graph=False)
        )

    with tf.device("/cpu:0"):
        train_data = dataset.train_data(hparams.batch_size)
        validation_data = dataset.validation_data(hparams.batch_size)

    with utils.get_distribution_scope(hparams.batch_size):
        if initial_epoch == 0:
            model = build_model(hparams, dataset)
            model.compile(
                optimizer=hparams.optimizer,
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
            )
        else:
            model = tf.keras.models.load_model(model_path)
            click.echo(f"Loaded model from epoch {initial_epoch}")

    lq.models.summary(model)

    model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=validation_data,
        validation_steps=dataset.validation_examples // hparams.batch_size,
        verbose=2 if tensorboard else 1,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    model.save_weights(path.join(output_dir, f"{build_model.__name__}_weights.h5"))


if __name__ == "__main__":
    import importlib

    # Running it without the CLI requires us to first import larq_zoo
    # in order to register the models and datasets
    importlib.import_module("larq_zoo")
    cli()
