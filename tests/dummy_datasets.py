import tensorflow_datasets as tfds
from zookeeper import Field, component
from zookeeper.tf import TFDSDataset


class TFDSDummyDatasetMixin:
    """
    This class is designed to be mixed-in with a subclass of
    `tfds.core.GeneratorBasedBuilder`, like so:

    ```
    class DummyOxfordFlowers102(
        TFDSDummyDatasetMixin,
        tfds.image.oxford_flowers102.OxfordFlowers102,
    ):
        pass
    ```

    This will make "dummy_oxford_flowers_102" a valid dataset known to TFDS. It
    will be identical to "oxford_flowers_102", except it will drop all splits
    other than "train", which will contain only the first `limit_examples`
    examples.

    After defining the example class above,
    `tfds.load("dummy_oxford_flowers_102:2.0.*", split="train")` will work and
    return valid data of exactly the same format returned by
    `tfds.load("oxford_flowers_102:2.0.*", split="train")`.
    """

    limit_examples = 4

    def _split_generators(self, dl_manager):
        for split_generator in super()._split_generators(dl_manager):  # type: ignore
            if split_generator.name == "train":
                return [split_generator]
        raise ValueError("Could not find SplitGenerator with name 'train'.")

    def _generate_examples(self, *args, **kwargs):
        for i, example in enumerate(
            super()._generate_examples(*args, **kwargs)  # type: ignore
        ):
            if i == self.limit_examples:
                break
            yield example


# Define a 'dummy' oxford flowers dataset, with 4 images.
class DummyOxfordFlowers102(TFDSDummyDatasetMixin, tfds.image.OxfordFlowers102):
    pass


@component
class DummyOxfordFlowers(TFDSDataset):
    name = Field("dummy_oxford_flowers102:2.0.*")
    train_split = Field("train")
    validation_split = Field("train")
    data_dir = Field("tests/fixtures/dummy_datasets")
