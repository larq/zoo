from typing import List, Optional

import tensorflow as tf
from cached_property import cached_property
from zookeeper import ComponentField, Field, factory

from core.model_factory import ModelFactory
from official.knowledge_distillation.teacher_student_losses import (
    AttentionMatchingLossLayer,
    OutputDistributionMatchingLossLayer,
    WeightedCrossEntropyLoss,
)


class KnowledgeDistillationModelFactory(ModelFactory):
    """This version of the ModelFactory blocks the input_tensor from being created.

    By using teacher and student models that subclass from this ModelFactory, the
    teacher and student models are forced to pick up the `input_tensor` from their
    parent `TeacherStudentModelFactory`. This way, the `input_tensor` is correctly
    shared among the teacher, student and teacher-student models.
    """

    input_tensor: tf.Tensor = Field()


def get_unique_layer_with_partial_name(
    model: tf.keras.Model, partial_name: str
) -> tf.keras.layers.Layer:
    """This helper function finds a layer in `model` that has `partial_name` in its name
    and returns it. If there is not exactly one match an error will be thrown.

    :returns Single keras layer from `model` that has `partial_name` in its `layer.name`
    :raises AssertionError when there is not exactly one layer found in `model` that has `partial_name`
        in its `layer.name`
    """
    results = [layer for layer in model.layers if partial_name in layer.name]
    assert (
        len(results) == 1
    ), f"Expected to find one layer matching {partial_name} in {model.name}, found {len(results)}"
    return results[0]


@factory
class TeacherStudentModelFactory(ModelFactory):
    """Model that contains two sub-models; a teacher network and a student network. The
    teacher model should be pre-trained and its knowledge can be transferred to the
    student in two ways:

    - Attention matching: the student is encouraged to pay attention to the same spatial locations at intermediate
        layers in the network.
    - Output distribution matching: the student is trained to match the (optionally softened) predictions of
        the teacher.

    Besides this, the student is also trained on the standard classification loss. All three losses can be weighted.
    """

    teacher_model: tf.keras.models.Model = ComponentField(allow_missing=True)
    student_model: tf.keras.models.Model = ComponentField()

    # Inherit this from the parent `MultiStageExperiment`.
    model_dir: str = Field()

    # Must be set if there is a teacher
    teacher_weights_path: str = Field(allow_missing=True)
    student_weights_path: str = Field(allow_missing=True)

    # parameters related to the standard cross-entropy training of the student on the target labels
    #  - weight on the loss component for standard classification
    classification_weight: float = Field(1.0)

    @cached_property
    def classification_loss(self):
        return WeightedCrossEntropyLoss(self.classification_weight)

    # parameters related to the training through attention matching between teacher and student activation volumes
    #  - weight on the loss component for spatial attention matching
    attention_matching_weight: float = Field(0.0)
    #  - list of partial names of the layers for which the outputs should be matched
    attention_matching_volume_names: Optional[List[str]] = Field(allow_missing=True)
    #  - allow teacher to be trained to better match activations with the student
    attention_matching_train_teacher: bool = Field(False)

    # parameters related to the training through the matching of the output predictions of the teacher and student
    #  - weight on the loss component for knowledge distillation
    output_matching_weight: float = Field(0.0)
    #  - temperature used for the softmax when matching distributions
    output_matching_softmax_temperature: float = Field(1.0)
    #  - allow the teacher to be trained during output distribution matching
    output_matching_train_teacher: bool = Field(False)

    def build(self) -> tf.keras.models.Model:
        def _load_submodel(sub_model: tf.keras.Model, path: str, name: str):
            try:
                print(f"Loading {name} weights from {path}.")
                sub_model.load_weights(
                    path
                ).expect_partial().assert_existing_objects_matched()
            except tf.errors.InvalidArgumentError as e:
                raise (
                    ValueError(
                        f"Could not find {name} weights at {path}: the directory seems to be wrong"
                    )
                ) from e
            except tf.errors.NotFoundError as e:
                raise (
                    ValueError(
                        f"Could not find {name} weights at {path}: the checkpoint files seem to be missing"
                    )
                ) from e

        if hasattr(self, "teacher_model"):
            assert hasattr(
                self, "teacher_weights_path"
            ), "Teachers should know probably know something, but no teacher_weights_path was provided."
            _load_submodel(
                self.teacher_model, path=self.teacher_weights_path, name="teacher"
            )

        if hasattr(self, "student_weights_path"):
            _load_submodel(
                self.student_model, path=self.student_weights_path, name="student"
            )

        if not hasattr(self, "teacher_model"):
            assert (
                self.output_matching_weight == 0 and self.attention_matching_weight == 0
            ), "No teacher set, but trying to use attention or distribution matching"
            # If there is no teacher model we do not need the teacher-student model and can instead simply return the student model
            return self.student_model
        else:
            assert (
                self.output_matching_weight > 0 or self.attention_matching_weight > 0
            ), "Teacher model loaded but all teacher-student knowledge distillation losses are 0"

        assert (
            self.teacher_model.input is self.student_model.input is self.input_tensor
        ), (
            "The teacher and the student must have the same `input_tensor`. "
            "Make sure that `input_tensor` is *not* set on the teacher or student models such that it is picked up "
            "from the teacher-student model. This can be done by making sure the factories for the teacher and "
            "student models subclass from `KnowledgeDistillationModelFactory` or by adding the field "
            "`input_tensor: tf.Tensor = Field()`."
        )

        # We take the output of the student and run it through some loss layers, which connects the
        # output to the teacher in the TF graph.
        combined_output = self.student_model.output

        if self.attention_matching_weight > 0:
            assert self.attention_matching_volume_names is not None
            tav, sav = [
                [
                    get_unique_layer_with_partial_name(model, name).output
                    for name in self.attention_matching_volume_names
                ]
                for model in (self.teacher_model, self.student_model)
            ]
            combined_output = AttentionMatchingLossLayer(
                loss_weight=self.attention_matching_weight,
                propagate_teacher_gradients=self.attention_matching_train_teacher,
            )([combined_output, tav, sav])

        if self.output_matching_weight > 0:
            tl, sl = [
                get_unique_layer_with_partial_name(model, "logits").output
                for model in (self.teacher_model, self.student_model)
            ]
            combined_output = OutputDistributionMatchingLossLayer(
                loss_weight=self.output_matching_weight,
                softmax_temperature=self.output_matching_softmax_temperature,
                propagate_teacher_gradients=self.output_matching_train_teacher,
            )([combined_output, tl, sl])

        combined_model = tf.keras.Model(
            inputs=self.input_tensor,
            outputs=combined_output,
            name="teacher_student_model",
        )

        # the classification loss is added when the model is compiled, as it depends on the targets
        return combined_model
