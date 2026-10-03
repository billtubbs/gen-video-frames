import numpy as np

from gen_video_frames.loops import (
    best_candidates,
    distance_band,
    seam_scores,
)


def brute_force_band(x, min_gap, max_gap):
    n = x.shape[0]
    band = np.full((max_gap - min_gap + 1, n), np.nan)
    for k, g in enumerate(range(min_gap, max_gap + 1)):
        for t in range(n - g):
            band[k, t] = ((x[t] - x[t + g]) ** 2).mean()
    return band


def test_distance_band_matches_brute_force(monkeypatch):
    # Small blocks so the block boundaries get exercised
    monkeypatch.setattr("gen_video_frames.loops.BLOCK_SIZE", 7)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(60, 12)).astype("float32")
    band = distance_band(x, 10, 25)
    expected = brute_force_band(x, 10, 25)
    assert np.array_equal(np.isnan(band), np.isnan(expected))
    ok = ~np.isnan(expected)
    assert np.allclose(band[ok], expected[ok], rtol=1e-4, atol=1e-2)


def test_seam_scores_window_rms():
    band = np.arange(20, dtype=float).reshape(2, 10) ** 2
    band[1, 8:] = np.nan
    scores = seam_scores(band, half_window=2)
    # t=4, row 0: frames 2..5
    assert np.isclose(scores[0, 4], np.sqrt(np.mean(band[0, 2:6])))
    # windows running off either end, or into NaN, are inf
    assert np.isinf(scores[0, 1])
    assert np.isinf(scores[1, 7])
    assert np.isfinite(scores[1, 6])


def test_finds_planted_loop():
    # Random-walk "video" with a repeat: frames 100.. copy frames 20..
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(200, 30)), axis=0).astype("float32")
    x[100:130] = x[20:50]
    band = distance_band(x, 70, 90)
    scores = seam_scores(band, half_window=3)
    (start, end, score), *_ = best_candidates(scores, 70, 3, 5)
    # Gap 80, with the whole +/-3 frame window inside the copied stretch
    assert end - start == 80
    assert 23 <= start <= 47
    assert score < 1e-3


def test_best_candidates_separation():
    scores = np.full((3, 50), np.inf)
    scores[0, 10] = 1.0
    scores[0, 11] = 1.1  # same loop shifted 1 frame - skipped
    scores[2, 30] = 2.0
    picked = best_candidates(
        scores, min_gap=20, n_candidates=5, min_separation=5
    )
    assert [(s, e) for s, e, _ in picked] == [(10, 30), (30, 52)]
