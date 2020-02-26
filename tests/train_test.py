import pytest
from click.testing import CliRunner

from larq_zoo import experiments
from tests import dummy_datasets  # noqa -- Needed to register the dummy dataset.


@pytest.mark.parametrize("command", list(experiments.cli.commands.keys()))
def test_cli(command):
    result = CliRunner().invoke(
        experiments.cli,
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
