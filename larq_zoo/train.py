from zookeeper import cli, build_train
from os import path
import click


@cli.command()
@click.option("--tensorboard/--no-tensorboard", default=True)
@build_train()
def train(build_model, dataset, hparams, output_dir, tensorboard):
    import larq as lq
    from larq_zoo import utils
    import tensorflow as tf

    initial_epoch = utils.get_current_epoch(output_dir)
    model_path = path.join(output_dir, "model")
    callbacks = [utils.ModelCheckpoint(filepath=model_path, save_weights_only=True)]
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
        model = build_model(hparams, dataset)
        model.compile(
            optimizer=hparams.optimizer,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
        )
        lq.models.summary(model)

        if initial_epoch > 0:
            model.load_weights(model_path)
            click.echo(f"Loaded model from epoch {initial_epoch}")

    model.fit(
        train_data,
        epochs=hparams.epochs,
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=validation_data,
        validation_steps=dataset.validation_examples // hparams.batch_size,
        verbose=2 if tensorboard else 1,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    model_name = build_model.__name__
    model.save_weights(path.join(output_dir, f"{model_name}_weights.h5"))

    # Save weights without top
    notop_model = build_model(hparams, dataset, include_top=False)
    notop_model.set_weights(model.get_weights()[: len(notop_model.get_weights())])
    notop_model.save_weights(path.join(output_dir, f"{model_name}_weights_notop.h5"))


if __name__ == "__main__":
    import importlib

    # Running it without the CLI requires us to first import larq_zoo
    # in order to register the models and datasets
    importlib.import_module("larq_zoo")
    cli()
