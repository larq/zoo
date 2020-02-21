from click.testing import CliRunner

from larq_zoo import experiments


def test_cli():
    result = CliRunner().invoke(
        experiments.cli,
        [
            "TrainBinaryAlexNet",
            "dataset=OxfordFlowers",
            "dataset.train_split='train'",
            "dataset.validation_split='validation'",
            "epochs=1",
            "batch_size=32",
            "--no-use_tensorboard",
            "--no-use_model_checkpointing",
        ],
    )
    assert result.exit_code == 0
