import pytest
from click.testing import CliRunner

from larq_zoo.training import basic_experiments
from tests import dummy_datasets  # noqa


@pytest.mark.parametrize("command", list(basic_experiments.cli.commands.keys()))
def test_cli(command):
    result = CliRunner().invoke(
        basic_experiments.cli,
        [
            command,
            "dataset=DummyOxfordFlowers",
            "epochs=1",
            "batch_size=2",
            "validation_frequency=5",
            "--no-use_tensorboard",
            "--no-use_model_checkpointing",
            "--no-save_weights",
        ],
    )
    assert result.exit_code == 0
