import pytest
from click.testing import CliRunner
from larq_zoo import train
import tensorflow as tf


@pytest.mark.skipif(
    int(tf.__version__[0]) == 2,
    reason="This currently fails on TF 2 due to https://github.com/larq/larq/pull/195.",
)
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
