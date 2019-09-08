from click.testing import CliRunner
from larq_zoo import train


def test_cli():
    runner = CliRunner()
    result = runner.invoke(
        train.cli,
        [
            "train",
            "binary_alexnet",
            "--dataset",
            "oxford_flowers102",
            "--hparams",
            "epochs=1,batch_size=230",
            "--no-tensorboard",
            "--validationset",
        ],
    )
    assert result.exit_code == 0
