from click.testing import CliRunner

import larq_zoo


def test_cli():
    result = CliRunner().invoke(
        larq_zoo.train.cli,
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
