from larq_flock import cli, build_train


@cli.command()
@build_train
def train(build_model, dataset, hparams, output_dir, epochs):
    pass


if __name__ == "__main__":
    import importlib

    # Running it without the CLI requires us to first import larq_zoo
    # in order to register the models and datasets
    importlib.import_module("larq_zoo")
    cli()
