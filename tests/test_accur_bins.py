"""Tests for fdup.utils.accur_bins (accur_bins.py)."""

from __future__ import annotations

import numpy as np
import pytest

from fdup.utils import accur_bins
from fdup.utils.accur_bins import _log_bins
from fdup.utils.tree import _make_seeds_dtype


def _seeds(acc) -> np.ndarray:
    """Minimal structured seeds array with the given ``acc`` values."""
    acc = np.asarray(acc, dtype=np.float64)
    seeds = np.zeros(len(acc), dtype=_make_seeds_dtype(np.dtype(np.float64)))
    seeds["acc"] = acc
    return seeds


# ---------------------------------------------------------------------------
# _log_bins
# ---------------------------------------------------------------------------


class TestLogBins:
    def test_example_depth(self):
        b = _log_bins(1234567)
        # k = -6: first cutoff floor(1234567 * 1e-6) == 1
        assert b[0] == 1
        assert b[-1] == 1234567
        assert len(b) == 6 * 9 + 1

    def test_exact_values_small(self):
        np.testing.assert_array_equal(
            _log_bins(25), [2, 5, 7, 10, 12, 15, 17, 20, 22, 25]
        )

    def test_power_of_ten(self):
        np.testing.assert_array_equal(
            _log_bins(100),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )

    def test_single_digit(self):
        np.testing.assert_array_equal(_log_bins(7), [7])

    def test_empty(self):
        assert _log_bins(0).size == 0

    @pytest.mark.parametrize("n", [1, 9, 10, 11, 99, 1000, 98765, 10**7 + 3])
    def test_strictly_increasing_and_bounded(self, n):
        b = _log_bins(n)
        assert np.all(np.diff(b) > 0)
        assert 1 <= b[0] <= 9
        assert b[-1] == n


# ---------------------------------------------------------------------------
# accur_bins
# ---------------------------------------------------------------------------


class TestAccurBins:
    def test_cumulative_weighted_average(self):
        acc = [4.0, 3.0, 2.0, 1.0]
        accurs = np.array([1.0, 0.0, 1.0, 0.5], dtype=np.float32)
        out, _ = accur_bins(accurs, _seeds(acc), bins=[1, 2, 4])
        np.testing.assert_allclose(out, [1.0, 4 / 7, 6.5 / 10])

    def test_non_cumulative(self):
        acc = [4.0, 3.0, 2.0, 1.0]
        accurs = np.array([1.0, 0.0, 1.0, 0.5], dtype=np.float32)
        out, _ = accur_bins(accurs, _seeds(acc), cumulative=False, bins=[1, 2, 4])
        np.testing.assert_allclose(out, [1.0, 0.0, 2.5 / 3])

    def test_default_bins_length(self):
        n = 250
        rng = np.random.default_rng(0)
        out, _ = accur_bins(rng.random(n), _seeds(np.arange(n, 0, -1)))
        assert out.shape == _log_bins(n).shape
        assert out.dtype == np.float64

    def test_last_cumulative_bin_is_global_average(self):
        rng = np.random.default_rng(1)
        n = 1234
        acc = rng.random(n) * 100
        accurs = rng.random(n)
        out, _ = accur_bins(accurs, _seeds(acc))
        np.testing.assert_allclose(out[-1], np.average(accurs, weights=acc))

    def test_ranked_by_acc_descending(self):
        # Same seeds in a shuffled order must give the same result.
        acc = np.array([4.0, 3.0, 2.0, 1.0])
        accurs = np.array([1.0, 0.0, 1.0, 0.5])
        perm = np.array([2, 0, 3, 1])
        a, _ = accur_bins(accurs, _seeds(acc), bins=[1, 2, 4])
        b, _ = accur_bins(accurs[perm], _seeds(acc[perm]), bins=[1, 2, 4])
        np.testing.assert_allclose(a, b)

    def test_degenerate_seeds_excluded(self):
        acc = [4.0, 3.0, 2.0]
        accurs = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        out, _ = accur_bins(accurs, _seeds(acc), cumulative=False, bins=[1, 2, 3])
        assert out[0] == 1.0
        assert np.isnan(out[1])
        assert out[2] == 0.0

    def test_integer_acc_dtype(self):
        seeds = np.zeros(3, dtype=_make_seeds_dtype(np.dtype(np.uint32)))
        seeds["acc"] = [30, 20, 10]
        out, _ = accur_bins(np.array([1.0, 0.0, 1.0]), seeds, bins=[3])
        np.testing.assert_allclose(out, [40 / 60])

    def test_returns_default_cutoffs(self):
        n = 250
        _, (b_abs, b_rel) = accur_bins(np.ones(n), _seeds(np.arange(n, 0, -1)))
        np.testing.assert_array_equal(b_abs, _log_bins(n))
        assert b_abs.dtype == np.int64
        assert b_rel.dtype == np.float64
        np.testing.assert_allclose(b_rel, b_abs / n)
        assert b_rel[-1] == 1.0

    def test_returns_custom_cutoffs(self):
        _, (b_abs, b_rel) = accur_bins(
            np.ones(4), _seeds([4.0, 3.0, 2.0, 1.0]), bins=[1, 2, 3]
        )
        np.testing.assert_array_equal(b_abs, [1, 2, 3])
        np.testing.assert_allclose(b_rel, [0.25, 0.5, 0.75])

    def test_empty_seeds(self):
        out, (b_abs, b_rel) = accur_bins(np.empty(0, dtype=np.float32), _seeds([]))
        assert out.shape == b_abs.shape == b_rel.shape == (0,)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            accur_bins(np.zeros(3), _seeds([1.0, 2.0]))

    def test_missing_acc_field_raises(self):
        with pytest.raises(ValueError, match="'acc'"):
            accur_bins(np.zeros(2), np.zeros(2))

    @pytest.mark.parametrize(
        "bins",
        [[], [[1, 2]], [0, 2], [1, 5], [2, 2], [3, 1], [1.5, 3]],
    )
    def test_bad_bins_raise(self, bins):
        with pytest.raises(ValueError):
            accur_bins(np.zeros(4), _seeds([4.0, 3.0, 2.0, 1.0]), bins=bins)
