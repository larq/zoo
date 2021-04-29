from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from zookeeper import ComponentField, Field

from larq_zoo.training.knowledge_distillation.knowledge_distillation import (
    TeacherStudentModelFactory,
)
from larq_zoo.training.train import TrainLarqZooModel


class TrainingPhase(TrainLarqZooModel):
    """Class used in multi-stage experiments with teacher-student knowledge distillation.

    This class makes it easy to run teacher-student knowledge distillation experiments
    as part of a sequence. After running the experiment, the student model weights will
    be saved.
    """

    # stage as part of a sequence of experiments, starting at 0
    stage: int = Field()

    model = ComponentField(TeacherStudentModelFactory)
    teacher_model: tf.keras.models.Model = ComponentField(allow_missing=True)
    # Can't really be missing but might be set on the teacher-student model directly
    student_model: tf.keras.models.Model = ComponentField(allow_missing=True)

    # Must be set if there is a teacher and allow_missing teacher weights is not True.
    # Either a full path or the name of a network (in which case it will be sought in the current `model_dir`).
    initialize_teacher_weights_from: str = Field(allow_missing=True)
    # Explicitly allow missing teacher weights (this should be an explicit decision, not an accident).
    allow_missing_teacher_weights: bool = Field(False)
    # Optional: initialize the student weights from here if set.
    initialize_student_weights_from: str = Field(allow_missing=True)

    # This will be inherited from the parent `MultiStageExperiment`.
    parent_output_dir: str = Field(allow_missing=True)

    @Field
    def output_dir(self) -> Path:
        """Main experiment output directory.

        In this directory, the training checkpoints, logs, and tensorboard files will be
        stored for this (sub-)experiment.
        """

        # When running as part of an `MultiStageExperiment` the outputs of this `stage` of the experiment will be
        # stored in a sub directory of the `MultiStageExperiment` which is named after the current stage index.
        if hasattr(self, "parent_output_dir"):
            return Path(self.parent_output_dir) / f"stage_{self.stage}"

        return (
            Path.home()
            / "zookeeper-logs"
            / self.dataset.__class__.__name__
            / self.__class__.__name__
            / datetime.now().strftime("%Y%m%d_%H%M")
        )

    @Field
    def model_dir(self) -> str:
        """The directory in which trained models are stored.

        When running as part of a sequence, this directory is shared among all stages
        such that the results of earlier stages can be used in later stages.
        """
        if hasattr(self, "parent_output_dir"):
            base = Path(self.parent_output_dir)
        else:
            base = Path(self.output_dir)
        return str(base / "models")

    def run(self):
        super().run()
        student = self.__base_getattribute__("model").student_model
        # Specifically saving the student at the end of training as this sub-model is what the training procedure
        # should optimize.
        student.save_weights(str(Path(self.model_dir) / student.name))


class LarqZooModelTrainingPhase(TrainingPhase):
    # parameters related to the standard cross-entropy training of the student on the target labels
    #  - weight on the loss component for standard classification
    classification_weight: float = Field(1.0)

    # parameters related to the training through attention matching between teacher and student activation volumes
    #  - weight on the loss component for spatial attention matching
    attention_matching_weight: float = Field(0.0)
    #  - list of partial names of the layers for which the outputs should be matched
    attention_matching_volume_names: Optional[List[str]] = Field(allow_missing=True)
    #  - optional separate list of partial names for the teacher. If not given, the names above will be used.
    attention_matching_volume_names_teacher: Optional[List[str]] = Field(
        allow_missing=True
    )
    #  - allow teacher to be trained to better match activations with the student
    attention_matching_train_teacher: bool = Field(False)

    # parameters related to the training through the matching of the output predictions of the teacher and student
    #  - weight on the loss component for knowledge distillation
    output_matching_weight: float = Field(0.0)
    #  - temperature used for the softmax when matching distributions
    output_matching_softmax_temperature: float = Field(1.0)
    #  - allow the teacher to be trained during output distribution matching
    output_matching_train_teacher: bool = Field(False)

    @Field
    def loss(self):
        return getattr(self.__base_getattribute__("model"), "classification_loss")

    metrics = Field(lambda: ["accuracy", "sparse_top_k_categorical_accuracy"])


class MultiStageExperiment:
    """Allows running a series of `KnowledgeDistillationExperiment`s in sequence."""

    initial_stage: int = Field(0)

    # To add a new stage, also increment the hard-coded `5` in the `experiments`
    # definition below.
    stage_0: TrainingPhase = ComponentField(allow_missing=True)
    stage_1: TrainingPhase = ComponentField(allow_missing=True)
    stage_2: TrainingPhase = ComponentField(allow_missing=True)
    stage_3: TrainingPhase = ComponentField(allow_missing=True)
    stage_4: TrainingPhase = ComponentField(allow_missing=True)

    @property
    def experiments(self):
        for i in range(5):
            exp = getattr(self, f"stage_{i}", None)
            if exp is not None:
                yield exp

    def __post_configure__(self):
        assert 0 <= self.initial_stage < 5

        # Check that all stages have the correct stage number (for setting
        # output directories, et cetera).
        for i, exp in enumerate(self.experiments):
            if exp is not None:
                assert exp.stage == i

        # Check that all stages being used are listed in sequence, without
        # `None` in between.
        for prev_exp, next_exp in zip(
            list(self.experiments)[self.initial_stage :],
            list(self.experiments)[self.initial_stage + 1 :],
        ):
            if prev_exp is None:
                assert next_exp is None

    @Field
    def parent_output_dir(self) -> str:
        """Top level experiment directory shared by all sub-experiments.
        This directory will have the following structure:
        ```
        parent_output_dir/models/  # dir shared among all experiments in the sequence to store trained models
        parent_output_dir/stage_0/  # dir with artifacts (checkpoints, logs, tensorboards, ...) of stage 0
        ...
        parent_output_dir/stage_n/ # dir with artifacts (checkpoints, logs, tensorboards, ...) of stage n
        ```
        """
        return str(
            Path.home()
            / "zookeeper-logs"
            / "knowledge_distillation"
            / self.__class__.__name__
            / datetime.now().strftime("%Y%m%d_%H%M")
        )

    @Field
    def model_dir(self) -> Path:
        """Directory shared by all sub-experiments where the models which have completed
        training are stored."""
        return Path(self.parent_output_dir) / "models"

    def run(self) -> None:
        Path(self.parent_output_dir).mkdir(parents=True, exist_ok=True)

        for experiment in self.experiments:
            if experiment.stage < self.initial_stage:
                continue
            print(f"Starting stage {experiment.stage} at {datetime.now().isoformat()}.")
            experiment.run()
