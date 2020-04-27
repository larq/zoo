from typing import Optional, Tuple

import tensorflow as tf
from zookeeper import ComponentField, Field
from zookeeper.tf import Dataset

from larq_zoo.core import utils

DimType = Optional[int]


class ModelFactory:
    """A base class for Larq Zoo models. Defines some common fields."""

    input_quantizer = None
    kernel_quantizer = None
    kernel_constraint = None

    # This field is included for automatic inference of `num_clases`, if no
    # value is otherwise provided. We set `allow_missing` because we don't want
    # to throw an error if a dataset is not provided, as long as `num_classes`
    # is overriden.
    dataset: Optional[Dataset] = ComponentField(allow_missing=True)

    @Field
    def num_classes(self) -> int:
        if self.dataset is None:
            raise TypeError(
                "No `dataset` is defined so unable to infer `num_classes`. Please "
                "provide a `dataset` or override `num_classes` directly."
            )
        return self.dataset.num_classes

    include_top: bool = Field(True)
    weights: Optional[str] = Field(None)

    input_shape: Optional[Tuple[DimType, DimType, DimType]] = Field(None)
    input_tensor: Optional[tf.Tensor] = Field(None)

    @property
    def image_input(self) -> tf.Tensor:
        if not hasattr(self, "_image_input"):
            input_shape = utils.validate_input(
                self.input_shape, self.weights, self.include_top, self.num_classes,
            )
            self._image_input = utils.get_input_layer(input_shape, self.input_tensor)
        return self._image_input
