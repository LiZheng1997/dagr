"""Unit tests for ipc/fit_transform.py. Run: python3 -m pytest ipc/test_fit_transform.py

Pure numpy, no torch, so it runs on any host.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_transform import make_fit  # noqa: E402

W, H = 320, 215


def test_dsec_matches_training_loader():
    tf = make_fit(640, 480, W, H, "dsec")
    # Training loader: keep y < 430, then halve.
    xs = np.array([0, 639, 100, 300])
    ys = np.array([0, 429, 430, 479])
    mask, mx, my = tf.event_mask_and_coords(xs, ys)
    assert mask.tolist() == [True, True, False, False]
    assert mx.tolist() == [0, 319]
    assert my.tolist() == [0, 214]
    # A model box maps back by 2x.
    back = tf.inv_box_xywh(np.array([[10, 20, 50, 60]]))
    assert np.allclose(back, [[20, 40, 80, 80]])


def test_letterbox_keeps_whole_frame():
    tf = make_fit(320, 320, W, H, "letterbox")
    s = 215 / 320
    assert np.isclose(tf.sx, s) and np.isclose(tf.sy, s)
    # Every sensor pixel, including the bottom row, survives.
    xs = np.array([0, 319, 160, 0, 319])
    ys = np.array([0, 0, 160, 319, 319])
    mask, mx, my = tf.event_mask_and_coords(xs, ys)
    assert mask.all()
    assert my.max() == 214 and my.min() == 0
    # Horizontal padding is symmetric: content spans [ox, ox + 215).
    assert mx.min() >= int(tf.ox) and mx.max() < int(tf.ox) + 215 + 1
    # Round trip of a box is exact up to float error.
    box_model = np.array([[tf.ox + 10, 20, tf.ox + 110, 120]])
    back = tf.inv_box_xywh(box_model)
    assert np.allclose(back, [[10 / s, 20 / s, 100 / s, 100 / s]])


def test_letterbox_clips_padding_to_sensor():
    tf = make_fit(320, 320, W, H, "letterbox")
    # A box that spills into the black side padding is clipped to the frame.
    back = tf.inv_box_xywh(np.array([[0, 0, W, H]]))
    assert np.allclose(back, [[0, 0, 320, 320]])


def test_squash_and_crop_top_shapes():
    sq = make_fit(320, 320, W, H, "squash")
    assert np.isclose(sq.sx, 1.0) and np.isclose(sq.sy, 215 / 320)
    mask, _, my = sq.event_mask_and_coords(np.array([5]), np.array([319]))
    assert mask.all() and my[0] == 214

    ct = make_fit(320, 320, W, H, "crop_top")
    mask, _, _ = ct.event_mask_and_coords(np.array([5, 5]), np.array([214, 215]))
    assert mask.tolist() == [True, False]  # legacy: bottom third dropped


def test_apply_image_letterbox_geometry():
    tf = make_fit(320, 320, W, H, "letterbox")
    img = np.full((320, 320, 3), 200, dtype=np.uint8)
    out = tf.apply_image(img)
    assert out.shape == (H, W, 3)
    ox = int(round(tf.ox))
    # Side padding black, centre filled.
    assert out[:, :ox - 1].max() == 0 and out[:, ox + 216:].max() == 0
    assert out[H // 2, W // 2].min() > 150


def test_apply_image_dsec_geometry():
    tf = make_fit(640, 480, W, H, "dsec")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:430] = 255          # rows that survive the crop
    out = tf.apply_image(img)
    assert out.shape == (H, W, 3)
    assert out.min() > 200   # nothing from the discarded rows leaks in


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
