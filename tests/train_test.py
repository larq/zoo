from unittest import mock

import pytest
import tensorflow_datasets as tfds
from click.testing import CliRunner
from zookeeper.tf.dataset import TFDSDataset

from larq_zoo.training import basic_experiments, multi_stage_experiments
from tests import dummy_datasets  # noqa


@pytest.mark.parametrize("command", list(basic_experiments.cli.commands.keys()))
@tfds.testing.mock_data(num_examples=2, data_dir="gs://tfds-data/dataset_info")
@mock.patch.object(TFDSDataset, "num_examples", return_value=2)
def test_cli(_, command):
    result = CliRunner().invoke(
        basic_experiments.cli,
        [
            command,
            "dataset=ImageNet",
            "epochs=1",
            "batch_size=2",
            "validation_frequency=5",
            "--no-use_tensorboard",
            "--no-use_model_checkpointing",
            "--no-save_weights",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


@pytest.mark.parametrize("command,phases", [("TrainR2BStrongBaseline", 2)])
def test_multi_stage_experiments(command, phases):
    arguments = [command] + ["--no-use_tensorboard", "--no-use_model_checkpointing"]
    for phase in range(phases):
        arguments.extend(
            [
                f"stage_{phase}.{c}"
                for c in [
                    "dataset=DummyOxfordFlowers",
                    "epochs=1",
                    "batch_size=2",
                    "validation_frequency=5",
                ]
            ]
        )

    result = CliRunner().invoke(multi_stage_experiments.cli, args=arguments)
    assert result.exit_code == 0
