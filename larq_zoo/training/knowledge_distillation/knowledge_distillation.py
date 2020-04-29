from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from zookeeper import ComponentField, Field, factory

from larq_zoo.core.model_factory import ModelFactory


class AttentionMatchingLossLayer(tf.keras.layers.Layer):
    """Layer that adds a loss and a metric based on the difference is spatial locations
    of the attention of activation volumes of a teacher and a student.

    This loss is described in Section 4.2 of
    `[Martinez et. al., Training binary neural networks with real-to-binary
    convolutions (2020)](https://openreview.net/forum?id=BJg4NgBKvH)`.
    """

    def __init__(
        self, loss_weight: float, propagate_teacher_gradients: bool = False, **kwargs
    ):
        """Creates a loss layer that computes a single scalar loss value based on the
        difference of attention of one or more pairs of activation values.

        After initializing the layer, it should be called on a list of length 3, where:
         - the first element is a tensor that will be returned as is,
         - the second a list of tensors of teacher activation volumes (or a single tensor),
         - the third element the student activation volume(s), that match those of the teacher.

        This allows the following usage:
        ```
        x = ...

        x = AttentionMatchingLossLayer(loss_weight=self.attention_matching_weight)(
            [
                x,
                [teacher.block1_output, teacher.block2_output],
                [student.block1_output, student.block2_output]
            ]
        )
        ```

        :param loss_weight: A scaling factor for the loss value.
        :param propagate_teacher_gradients: Propagate gradients from this loss back into the teacher network.
        """
        super().__init__(name="attention_matching_loss", **kwargs)
        self.propagate_teacher_gradients = propagate_teacher_gradients
        self.loss_weight = loss_weight

    @staticmethod
    def _normalized_attention_vector(activation_volume: tf.Tensor) -> tf.Tensor:
        attention_area = tf.reduce_mean(tf.math.square(activation_volume), axis=-1)
        batch_size = tf.shape(activation_volume)[0]
        attention_vector = tf.reshape(attention_area, (batch_size, -1))
        return attention_vector / tf.reduce_max(
            attention_vector, axis=-1, keepdims=True
        )

    @staticmethod
    def _layer_attention_loss(
        student_activation_volume: tf.Tensor, teacher_activation_volume: tf.Tensor
    ) -> tf.Tensor:
        if not (
            len(student_activation_volume.shape) == 4
            and len(teacher_activation_volume.shape) == 4
        ):
            raise NotImplementedError(
                "Attention matching is only implemented for 4d activation volumes ([B x H/W x W/H x C]). "
                f"However the received input tensors had shapes:\n "
                f" - student: {student_activation_volume.shape}\n"
                f" - teacher: {teacher_activation_volume.shape}."
            )

        qs = AttentionMatchingLossLayer._normalized_attention_vector(
            student_activation_volume
        )
        qt = AttentionMatchingLossLayer._normalized_attention_vector(
            teacher_activation_volume
        )

        return tf.reduce_mean(tf.math.square(qs - qt), axis=-1)

    @staticmethod
    def _process_inputs(call_inputs):
        _basic_usage = (
            "A list with three inputs is expected;\n"
            " - the first being a tensor that will be returned as is,\n"
            " - the second being a list of the teacher activation volumes or a single teacher activation volume tensor,\n"
            " - and the third being (a list of) the student activation volume(s) matching those of the teacher."
        )
        if isinstance(call_inputs, list):
            if len(call_inputs) == 3:
                _basic_usage = (
                    "AttentionMatchingLossLayer expects the second and third elements of the input to be "
                    "lists of activation volumes (or single tensors) with identical dimensions"
                )
                # Will turn a single tensor into a list, keep a list untouched
                teacher_activation_volumes = list(call_inputs[1])
                student_activation_volumes = list(call_inputs[2])
                if len(teacher_activation_volumes) != len(student_activation_volumes):
                    raise ValueError(
                        f"{_basic_usage}\n\nIt was instead called with arguments of different lengths:\n"
                        f" - teacher_activation_volumes had length {len(teacher_activation_volumes)}, \n"
                        f" - student_activation_volumes had length {len(student_activation_volumes)}."
                    )
                bad_input = False
                description = ""
                for idx, (tv, sv) in enumerate(
                    zip(teacher_activation_volumes, student_activation_volumes)
                ):
                    if tf.is_tensor(tv) and tf.is_tensor(sv):
                        description += f"\n {idx} - teacher: Tensor(shape={tv.shape}) student: Tensor(shape={sv.shape})"
                        # checking str representation as it is robust with and without a first None dimension
                        if str(tv.shape) != str(sv.shape):
                            bad_input = True
                    else:
                        bad_input = True
                        description += f"\n {idx} - teacher type: {type(tv)} student type: {type(sv)}\n"
                if bad_input:
                    raise ValueError(
                        _basic_usage + f"\n\nIt was instead called with:\n{description}"
                    )
            else:
                raise ValueError(
                    f"AttentionMatchingLossLayer was called with a list of inputs of size {len(call_inputs)}. "
                    + _basic_usage
                )
        else:
            raise ValueError(
                f"AttentionMatchingLossLayer was called with a {type(call_inputs)}, "
                f"while a list was expected. Specifically: \n\n" + _basic_usage
            )
        return call_inputs[0], teacher_activation_volumes, student_activation_volumes

    def call(self, inputs: list) -> tf.Tensor:
        input_tensor, teacher_volumes, student_volumes = self._process_inputs(inputs)

        if not self.propagate_teacher_gradients:
            teacher_volumes = [
                tf.keras.backend.stop_gradient(tav) for tav in teacher_volumes
            ]

        layer_losses = [
            self._layer_attention_loss(sa, ta)
            for sa, ta in zip(student_volumes, teacher_volumes)
        ]
        # Keras models can add  regularization losses which are shape=(); already reduced over the batch dimension.
        # This means the loss returned here should also be of shape (), otherwise the keras reduction logic fails.
        loss = self.loss_weight * tf.reduce_mean(tf.stack(layer_losses, axis=-1))

        self.add_loss(loss, inputs=True)
        self.add_metric(loss, name="attention_loss", aggregation="mean")

        return input_tensor

    def get_config(self):
        return {
            "loss_weight": self.loss_weight,
            "propagate_teacher_gradients": self.propagate_teacher_gradients,
        }


class OutputDistributionMatchingLossLayer(tf.keras.layers.Layer):
    """Layer that adds a loss and a metric based on the difference between the
    classification predictions of a teacher and a student.

    This loss was described in `[Distilling the Knowledge in a Neural Network, Hinton
    et. al., (2015)](https://arxiv.org/abs/1503.02531)` as (one form of) knowledge
    distillation.
    """

    def __init__(
        self,
        loss_weight: float,
        softmax_temperature: float,
        propagate_teacher_gradients: bool = True,
        **kwargs,
    ):
        """Create a loss layer that computes a single scalar loss value based on the
        difference of output prediction distributions of a teacher and a student.

        After initializing the layer, it should be called on a list of length 3, where:
         - the first element contains a tensor that will be returned as is,
         - the second element contains the logits of the teacher,
         - and the third the logits of the student.

        This allows the following usage, where a loss and metric will be added:
        ```
        teacher_model = ...
        student_model = ...

        x = AttentionMatchingLossLayer(loss_weight=self.attention_matching_weight)(
            [x, teacher.logits, student.logits]
        )
        ```
        :param loss_weight:  A scaling factor for the loss value.
        :param softmax_temperature: Temperature of the softmax. See [Distilling the Knowledge in a Neural Network,
            Hinton et. al., (2015)](https://arxiv.org/abs/1503.02531) for a description.
            A temperature of 1 trains the student to reproduce the original distribution of the teacher,
            while higher values soften the distribution.
        :param propagate_teacher_gradients: Propagate gradients from this loss back into the teacher network.
        """
        super().__init__(name="output_distribution_matching_loss", **kwargs)
        self.loss_weight = loss_weight
        self.softmax_temperature = softmax_temperature
        self.propagate_teacher_gradients = propagate_teacher_gradients

    @staticmethod
    def _process_inputs(call_inputs):
        _basic_usage = (
            "OutputDistributionMatchingLossLayer should be called with a list of length 3, where:\n "
            " - the first element contains a tensor that will be returned as is,\n"
            " - the second argument contains the logits of the teacher,\n"
            " - and the third the logits of the student.\n\n Instead, it was called with: "
        )
        if isinstance(call_inputs, list):
            if len(call_inputs) == 3 and all([tf.is_tensor(ci) for ci in call_inputs]):
                return tuple(call_inputs)
            raise ValueError(_basic_usage + str([type(ci) for ci in call_inputs]))
        else:
            raise ValueError(_basic_usage + str(type(call_inputs)))

    def call(self, inputs: list, **kwargs) -> tf.Tensor:
        input_tensor, teacher_logits, student_logits = self._process_inputs(inputs)
        if not self.propagate_teacher_gradients:
            teacher_logits = tf.stop_gradient(teacher_logits)
        # Keras models can add  regularization losses which are shape=(); already reduced over the batch dimension.
        # This means the loss returned here should also be of shape (), otherwise the keras reduction logic fails.
        loss = tf.reduce_mean(
            self.loss_weight
            * self.softmax_temperature ** 2
            * tf.keras.losses.kullback_leibler_divergence(
                y_true=tf.math.softmax(teacher_logits / self.softmax_temperature),
                y_pred=tf.math.softmax(student_logits / self.softmax_temperature),
            )
        )

        self.add_loss(loss, inputs=True)
        self.add_metric(loss, name="output_matching_loss", aggregation="mean")

        return input_tensor

    def get_config(self):
        return {
            "loss_weight": self.loss_weight,
            "softmax_temperature": self.softmax_temperature,
            "propagate_teacher_gradients": self.propagate_teacher_gradients,
        }


class WeightedCrossEntropyLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    """Layer that returns a weighted SparseCategoricalCrossentropy loss."""

    def __init__(self, loss_weight: float):
        super().__init__()
        self.loss_weight = loss_weight

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.loss_weight * super().__call__(y_true, y_pred)

    def get_config(self):
        return {"loss_weight": self.loss_weight}


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

    # Must be set if there is a teacher and allow_missing teacher weights is not True.
    # Either a full path or the name of a network (in which case it will be sought in the current `model_dir`).
    initialize_teacher_weights_from: str = Field(allow_missing=True)
    # explicitly allow missing teacher weights (this should be ane explicit decision, not an accident).
    allow_missing_teacher_weights: bool = Field()
    # Optional: initialize the student weights from here if set.
    initialize_student_weights_from: str = Field(allow_missing=True)
    # Optionally picked up from the parent experiment, used to prepend the arguments above when only a name of a
    # network is given rather than a full path
    model_dir: str = Field(allow_missing=True)

    # parameters related to the standard cross-entropy training of the student on the target labels
    #  - weight on the loss component for standard classification
    classification_weight: float = Field()

    _classification_loss = None

    @property
    def classification_loss(self) -> tf.keras.losses.Loss:
        if self._classification_loss is None:
            self._classification_loss = WeightedCrossEntropyLoss(
                self.classification_weight
            )
        return self._classification_loss

    # parameters related to the training through attention matching between teacher and student activation volumes
    #  - weight on the loss component for spatial attention matching
    attention_matching_weight: float = Field()
    #  - list of partial names of the layers for which the outputs should be matched
    attention_matching_volume_names: Optional[List[str]] = Field(allow_missing=True)
    #  - optional separate list of partial names for the teacher. If not given, the names above will be used.
    attention_matching_volume_names_teacher: Optional[List[str]] = Field(
        allow_missing=True
    )
    #  - allow teacher to be trained to better match activations with the student
    attention_matching_train_teacher: bool = Field()

    # parameters related to the training through the matching of the output predictions of the teacher and student
    #  - weight on the loss component for knowledge distillation
    output_matching_weight: float = Field()
    #  - temperature used for the softmax when matching distributions
    output_matching_softmax_temperature: float = Field()
    #  - allow the teacher to be trained during output distribution matching
    output_matching_train_teacher: bool = Field()

    def build(self) -> tf.keras.models.Model:
        def _load_submodel(sub_model: tf.keras.Model, path: str, name: str):
            if len(Path(path).parts) < 2:  # not a path but a network name
                path = str(Path(self.model_dir) / path)
            try:
                print(f"Loading {name} weights from {path}.")
                sub_model.load_weights(
                    path
                ).expect_partial()  # .assert_existing_objects_matched()
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
            if hasattr(self, "initialize_teacher_weights_from"):
                _load_submodel(
                    self.teacher_model,
                    path=self.initialize_teacher_weights_from,
                    name="teacher",
                )
            else:
                if not self.allow_missing_teacher_weights:
                    raise ValueError(
                        "Teachers should know probably know something, but no teacher_weights_path was provided."
                    )

        if hasattr(self, "initialize_student_weights_from"):
            _load_submodel(
                self.student_model,
                path=self.initialize_student_weights_from,
                name="student",
            )

        if not hasattr(self, "teacher_model"):
            assert (
                self.output_matching_weight == 0 and self.attention_matching_weight == 0
            ), "No teacher set, but trying to use attention or distribution matching"
            # If there is no teacher model we do not need the teacher-student model
            # and can instead simply return the student model
            return self.student_model
        else:
            assert (
                self.output_matching_weight > 0 or self.attention_matching_weight > 0
            ), "Teacher model loaded but all teacher-student knowledge distillation losses are 0"

        assert (
            len(self.teacher_model.inputs) == 1 and len(self.student_model.inputs) == 1
        ), (
            "TeacherStudentModelFactory expects the teacher and student models to have one input each, but received:"
            f"\n - a teacher with {len(self.teacher_model.inputs)} inputs and "
            f"\n - a student with {len(self.student_model.inputs)} inputs. "
        )

        # We take the output of the student and run it through some loss layers, which connects the
        # output to the teacher in the TF graph.
        combined_output = self.student_model.output

        if self.attention_matching_weight > 0:
            assert self.attention_matching_volume_names is not None
            attention_volume_names_teacher = (
                self.attention_matching_volume_names_teacher
                if hasattr(self, "attention_matching_volume_names_teacher")
                else self.attention_matching_volume_names
            )
            teacher_attention_volumes = [
                get_unique_layer_with_partial_name(self.teacher_model, name).output
                for name in attention_volume_names_teacher
            ]
            student_attention_volumes = [
                get_unique_layer_with_partial_name(self.student_model, name).output
                for name in self.attention_matching_volume_names
            ]

            combined_output = AttentionMatchingLossLayer(
                loss_weight=self.attention_matching_weight,
                propagate_teacher_gradients=self.attention_matching_train_teacher,
            )([combined_output, teacher_attention_volumes, student_attention_volumes])

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

        if (
            not self.attention_matching_train_teacher
            and not self.output_matching_train_teacher
        ):
            for layer in self.teacher_model.layers:
                layer.trainable = False

        combined_model = tf.keras.models.Model(
            inputs=[*self.teacher_model.inputs, *self.student_model.inputs],
            outputs=combined_output,
            name="combined_model",
        )

        # the classification loss is added when the model is compiled, as it depends on the targets
        # Return a model which takes a single input and passes it to both the teacher and the student.
        return tf.keras.Model(
            inputs=self.image_input,
            outputs=combined_model([self.image_input, self.image_input]),
            name="teacher_student_model",
        )
