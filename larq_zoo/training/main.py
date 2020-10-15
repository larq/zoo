import importlib


def cli():
    for experiments_file in (
        "larq_zoo.training.basic_experiments",
        "larq_zoo.training.multi_stage_experiments",
        "larq_zoo.training.sota_experiments",
    ):
        importlib.import_module(experiments_file)

    from zookeeper import cli

    cli()


if __name__ == "__main__":
    cli()
