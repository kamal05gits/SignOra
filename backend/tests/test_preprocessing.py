# ---------------------------------------------------------------------------
# Preprocessing unit tests (no ML framework required)
# ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest
from app.preprocessing import (
    PreprocessingError,
    decode_image,
    frames_to_sequence,
    landmarks_matrix,
    normalize_landmarks,
    pick_hand,
    prepare_image,
    softmax,
    top_k_indices,
)
from app.schemas import Frame, Hand, Landmark
from conftest import png_base64, synthetic_hand


def _frame(*handedness: str) -> Frame:
    return Frame(
        hands=[
            Hand(
                landmarks=[Landmark(**point) for point in synthetic_hand(seed=i)["landmarks"]],
                handedness=hand or "Right",
            )
            for i, hand in enumerate(handedness)
        ]
    )


def test_normalize_landmarks_makes_the_hand_position_and_scale_invariant():
    hand = Hand(**synthetic_hand(seed=7))
    points = landmarks_matrix(hand)

    moved = points + np.asarray([0.3, -0.2, 0.05], dtype=np.float32)
    scaled = points * 2.5

    base = normalize_landmarks(points, "wrist")
    assert np.allclose(normalize_landmarks(moved, "wrist"), base, atol=1e-5)
    assert np.allclose(normalize_landmarks(scaled, "wrist"), base, atol=1e-5)
    # The wrist lands on the origin and the wrist→middle-MCP length is 1.
    assert np.allclose(base[0], 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(base[9]), 1.0, atol=1e-6)


def test_normalize_landmarks_none_passes_through():
    points = landmarks_matrix(Hand(**synthetic_hand(seed=2)))
    assert np.allclose(normalize_landmarks(points, "none"), points)


def test_frames_to_sequence_pads_and_truncates():
    frames = [_frame("Right") for _ in range(4)]

    padded, filled = frames_to_sequence(frames, length=8)
    assert padded.shape == (1, 8, 63)
    assert filled == 4
    # Left padding is zeros, real frames come last.
    assert np.all(padded[0, 0] == 0)
    assert not np.all(padded[0, -1] == 0)

    truncated, filled = frames_to_sequence(frames, length=2)
    assert truncated.shape == (1, 2, 63)
    assert filled == 4


def test_frames_to_sequence_counts_missing_hands():
    frames = [_frame("Right"), Frame(hands=[]), _frame("Right")]
    tensor, filled = frames_to_sequence(frames, length=3)
    assert tensor.shape == (1, 3, 63)
    assert filled == 2
    assert np.all(tensor[0, 1] == 0)


def test_include_handedness_appends_one_column():
    tensor, _ = frames_to_sequence([_frame("Right")], length=2, include_handedness=True)
    assert tensor.shape == (1, 2, 64)
    assert tensor[0, -1, -1] == 1.0


def test_pick_hand_modes():
    frame = _frame("Left", "Right")
    assert pick_hand(frame, "first").handedness == "Left"
    assert pick_hand(frame, "right").handedness == "Right"
    assert pick_hand(frame, "left").handedness == "Left"
    assert pick_hand(Frame(hands=[]), "first") is None


def test_decode_image_accepts_a_data_url_prefix():
    payload = f"data:image/png;base64,{png_base64(32)}"
    array = decode_image(payload)
    assert array.shape == (32, 32, 3)
    assert array.dtype == np.uint8


def test_decode_image_rejects_garbage():
    with pytest.raises(PreprocessingError):
        decode_image("%%%% not base64 %%%%")
    with pytest.raises(PreprocessingError):
        decode_image("")


def test_prepare_image_resizes_and_scales():
    array = np.full((20, 40, 3), 255, dtype=np.uint8)
    tensor = prepare_image(array, size=64)
    assert tensor.shape == (1, 64, 64, 3)
    assert np.allclose(tensor, 1.0)

    channels_first = prepare_image(array, size=64, data_format="channels_first")
    assert channels_first.shape == (1, 3, 64, 64)

    bgr = prepare_image(array, size=8, channel_order="BGR")
    assert bgr.shape == (1, 8, 8, 3)


def test_prepare_image_applies_mean_and_std():
    array = np.full((8, 8, 3), 128, dtype=np.uint8)
    tensor = prepare_image(array, size=8, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    expected = (128 / 255 - 0.5) / 0.25
    assert np.allclose(tensor, expected, atol=1e-3)


def test_softmax_and_top_k():
    scores = softmax(np.asarray([1.0, 2.0, 3.0]))
    assert np.isclose(scores.sum(), 1.0)
    assert scores[2] > scores[0]

    ranked = top_k_indices(np.asarray([0.1, 0.7, 0.2]), k=2)
    assert [index for index, _ in ranked] == [1, 2]
    assert ranked[0][1] == pytest.approx(0.7)
