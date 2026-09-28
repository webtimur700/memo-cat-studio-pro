"""Логика scripts/regress_video_variants.py: сопоставление моментов с контролем и проверка ориентации кадра."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import make_test_videos  # noqa: E402
import regress_video_variants as regress  # noqa: E402


def test_moments_match_by_overlap_not_by_exact_boundaries():
    control = [(40.2, 61.5), (74.82, 94.07), (135.0, 153.88)]
    variant = [(40.2, 62.27), (72.2, 94.91), (164.98, 179.98)]     # VFR «ночной»: два совпали, третий другой
    pairs = regress.match_moments(control, variant)
    assert [j for _, j, _ in pairs] == [0, 1, -1]
    assert pairs[0][2] > 0.9 and pairs[2][2] == 0.0


def test_iou():
    assert regress.iou((0, 10), (0, 10)) == 1.0
    assert regress.iou((0, 10), (5, 15)) == 5 / 15
    assert regress.iou((0, 10), (20, 30)) == 0.0


def test_orientation_prefers_upright_over_rotated_frames():
    rng = np.random.default_rng(1)
    frame = rng.random((regress.THUMB_H, regress.THUMB_W)).astype(np.float32)
    frame[:20] += 2.0                                       # асимметрия сверху-вниз, как у настоящего кадра
    noisy = frame + 0.1 * rng.random(frame.shape).astype(np.float32)
    assert regress.corr(noisy, frame) > 0.9
    assert regress.corr(noisy, frame[::-1, ::-1]) < regress.corr(noisy, frame)
    scores = {deg: regress.corr(noisy, c) for deg, c in regress.rotations(frame).items()}
    assert max(scores, key=scores.get) == 0
    turned = {deg: regress.corr(np.ascontiguousarray(regress.rotations(noisy)[90]), c) for deg, c in regress.rotations(frame).items()}
    assert max(turned, key=turned.get) == 90                # кадр, повёрнутый на 90°, узнаётся как повёрнутый


def test_generator_defines_all_variants_incl_rotations_and_hdr():
    names = set(make_test_videos.variants(60, 180.0))
    assert {"cfr_control.mp4", "vfr_random.mp4", "vfr_lowlight.mp4", "uhd4k30_hevc.mp4", "uhd4k60_h264.mp4",
            "hibitrate.mp4", "rot_m90.mp4", "rot_p90.mp4", "rot180.mp4", "hdr_hlg.mp4", "hdr_pq.mp4"} <= names
    assert {deg for deg, _ in make_test_videos.ROTATIONS.values()} == {-90, 90, 180}
    lowlight = make_test_videos.variants(30, 90.0)["vfr_lowlight.mp4"][1]
    assert "between(t,30.0,60.0)" in lowlight               # «ночной» участок — средняя треть любого исходника
