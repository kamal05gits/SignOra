# ---------------------------------------------------------------------------
# MODULE: Dataset evaluation
#
# Runs a trained model over the samples recorded in the browser's Dataset
# Collection page and computes real precision / recall / F1 plus a confusion
# matrix. The maths deliberately mirrors src/lib/evaluate.ts so the numbers on
# the Model page mean the same thing whether they came from the on-device
# heuristic classifier or from a trained model served by this backend.
# ---------------------------------------------------------------------------
from __future__ import annotations

import time

import numpy as np

from . import preprocessing
from .runtimes.base import ModelHandle
from .schemas import ClassMetrics, DatasetEvaluateRequest, EvaluationReport

REJECTED = "REJECTED"


def _predict_one(handle: ModelHandle, sample) -> tuple[str, float]:
    spec = handle.spec
    if sample.frames:
        tensor, _filled, _length = preprocessing.landmark_tensor(
            sample.frames,
            input_shape=handle.input_shape,
            sequence_length=spec.sequence_length,
            normalization=spec.normalization,
            include_handedness=spec.include_handedness,
        )
    elif sample.features:
        tensor = preprocessing.features_to_vector(sample.features, normalization=spec.normalization)
    else:
        raise ValueError(f"sample '{sample.id}' carries neither frames nor features")

    raw = np.asarray(handle.predict(tensor), dtype=np.float64).reshape(-1)
    scores = raw if handle.already_probabilistic else preprocessing.softmax(raw)
    index = int(np.argmax(scores))
    label, _gloss = spec.label_at(index)
    return label, float(scores[index])


def evaluate(handle: ModelHandle, request: DatasetEvaluateRequest) -> EvaluationReport:
    started = time.perf_counter()

    pairs: list[tuple[str, str]] = []
    rejected = 0
    for sample in request.samples:
        try:
            predicted, score = _predict_one(handle, sample)
        except ValueError:
            continue
        if score < request.confidence_threshold:
            predicted = REJECTED
            rejected += 1
        pairs.append((sample.label, predicted))

    if not pairs:
        return EvaluationReport(
            model_id=handle.spec.id,
            total_samples=0,
            correct=0,
            accuracy=0.0,
            rejected=rejected,
            classes=[],
            confusion_matrix=[],
            per_class=[],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    classes = sorted({label for pair in pairs for label in pair})
    index = {label: i for i, label in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    correct = 0
    for truth, predicted in pairs:
        matrix[index[truth]][index[predicted]] += 1
        if truth == predicted:
            correct += 1

    per_class: list[ClassMetrics] = []
    for label in classes:
        if label == REJECTED:
            continue
        i = index[label]
        support = sum(matrix[i])
        true_positive = matrix[i][i]
        predicted_positive = sum(matrix[j][i] for j in range(len(classes)))
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append(
            ClassMetrics(
                label=label,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                support=support,
            )
        )

    return EvaluationReport(
        model_id=handle.spec.id,
        total_samples=len(pairs),
        correct=correct,
        accuracy=round(correct / len(pairs), 4),
        rejected=rejected,
        classes=classes,
        confusion_matrix=matrix,
        per_class=per_class,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
