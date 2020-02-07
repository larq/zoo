from typing import Callable, Optional, Tuple, Union

import tensorflow as tf
from zookeeper import ComponentField, Field
from zookeeper.tf import Dataset

from larq_zoo import utils

# FIXME
QuantizerType = Union[
    tf.keras.layers.Layer, Callable[[tf.Tensor], tf.Tensor], str, None
]
ContraintType = Union[tf.keras.constraints.Constraint, str, None]
DimType = Optional[int]


class ModelFactory:
    """A base class for Larq Zoo models. Defines some common fields."""

    # Don't set any defaults here.
    input_quantizer: QuantizerType = Field()
    kernel_quantizer: QuantizerType = Field()
    kernel_constraint: ContraintType = Field()

    # This field is included for automatic inference of `num_clases`, if no
    # value is otherwise provided.
    dataset: Optional[Dataset] = ComponentField(None)

    @Field
    def num_classes(self) -> int:
        if self.dataset is None:
            raise TypeError("Must override either `dataset` or `num_classes`.")
        return self.dataset.num_classes

    input_shape: Optional[Tuple[DimType, DimType, DimType]] = Field(None)
    input_tensor: Optional[tf.Tensor] = Field(None)
    include_top: bool = Field(True)
    weights: Optional[str] = Field(None)

    @property
    def image_input(self) -> tf.Tensor:
        input_shape = utils.validate_input(
            self.input_shape, self.weights, self.include_top, self.num_classes
        )
        return utils.get_input_layer(input_shape, self.input_tensor)
