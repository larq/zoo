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
            "epochs=5",
            "batch_size=2",
            "validation_frequency=5",
            "--no-use_tensorboard",
            "--no-use_model_checkpointing",
            "--no-save_weights",
        ],
    )
    assert result.exit_code == 0
    # The dataset has four images in a single split, that is used for both train
    # and validation. Make sure we can train to the point that we get at least
    # three out of the four images correct.
    assert (
        "sparse_categorical_accuracy: 1.00" in result.output
        or "sparse_categorical_accuracy: 0.75" in result.output
    )
