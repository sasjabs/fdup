"""Tests for fdup.plots helpers."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import LineString, box

from fdup.plots import accurs_plot, huac_scatterplot, upscaled_map, windrose_plot


def _accurs_inputs():
    bins = np.array([0.01, 0.1, 1.0])
    acc = np.array([0.2, 0.5, 0.9])
    return acc, bins


def _huac_table():
    fine = np.array([1.0, 10.0, 100.0, 1000.0])
    coarse = fine * np.array([1.0, 1.1, 0.9, 1.2])
    return pd.DataFrame({
        "coarse_upstream_area": coarse,
        "fine_max_upstream_area": fine,
        "relative_upstream_area_error": (coarse - fine) / fine,
    })


def _windrose():
    return np.array([0.147, 0.103, 0.147, 0.103, 0.147, 0.103, 0.147, 0.103])


def _watershed():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")


def _network():
    return gpd.GeoDataFrame(
        geometry=[LineString([(0.0, 0.0), (1.0, 1.0)])],
        crs="EPSG:4326",
    )


def _assert_figure(fig, ax):
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def _assert_png(path):
    assert path.is_file()
    assert path.stat().st_size > 0


class TestAccursPlot:
    def test_saves_png(self, tmp_path):
        path = tmp_path / "accurs_plot.png"
        acc, bins = _accurs_inputs()
        fig, ax = accurs_plot(acc, bins, output=path)
        try:
            _assert_figure(fig, ax)
            _assert_png(path)
            assert ax.get_xscale() == "log"
            assert ax.get_ylim() == (0.0, 1.0)
        finally:
            plt.close(fig)

    def test_output_none_skips_default_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        acc, bins = _accurs_inputs()
        fig, ax = accurs_plot(acc, bins, output=None)
        try:
            _assert_figure(fig, ax)
            assert not (tmp_path / "accurs_plot.png").exists()
        finally:
            plt.close(fig)

    def test_mismatched_curve_lengths(self):
        with pytest.raises(ValueError):
            accurs_plot(
                np.array([0.2, 0.5]),
                np.array([0.01, 0.1, 1.0]),
                output=None,
            )

    def test_default_axis_labels(self):
        acc, bins = _accurs_inputs()
        fig, ax = accurs_plot(acc, bins, output=None)
        try:
            assert ax.get_xlabel() == "Fraction of rivers"
            assert ax.get_ylabel() == "Weighted accuracy score"
        finally:
            plt.close(fig)

    def test_custom_axis_labels(self):
        acc, bins = _accurs_inputs()
        fig, ax = accurs_plot(
            acc,
            bins,
            xlabel="Number of rivers",
            ylabel="Score",
            output=None,
        )
        try:
            assert ax.get_xlabel() == "Number of rivers"
            assert ax.get_ylabel() == "Score"
        finally:
            plt.close(fig)

    def test_xlim_matches_bin_range(self):
        acc, bins = _accurs_inputs()
        fig, ax = accurs_plot(acc, bins, output=None)
        try:
            assert ax.get_xlim() == (float(np.min(bins)), float(np.max(bins)))
        finally:
            plt.close(fig)

    def test_yrange_applied(self):
        acc, bins = _accurs_inputs()
        fig, ax = accurs_plot(acc, bins, yrange=(0.4, 0.9), output=None)
        try:
            assert ax.get_ylim() == (0.4, 0.9)
        finally:
            plt.close(fig)

    @pytest.mark.parametrize(
        "yrange",
        [(0.5,), (0.4, 0.9, 1.0), "0.4,0.9", (0.5, 0.5), ("a", "b")],
    )
    def test_invalid_yrange_raises(self, yrange):
        acc, bins = _accurs_inputs()
        with pytest.raises(ValueError):
            accurs_plot(acc, bins, yrange=yrange, output=None)


class TestHuacScatterplot:
    def test_saves_png(self, tmp_path):
        path = tmp_path / "huac_scatterplot.png"
        fig, ax = huac_scatterplot(_huac_table(), output=path)
        try:
            _assert_figure(fig, ax)
            _assert_png(path)
            assert ax.get_xscale() == "log"
            assert ax.get_yscale() == "log"
        finally:
            plt.close(fig)

    def test_output_none_skips_default_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fig, ax = huac_scatterplot(_huac_table(), output=None)
        try:
            _assert_figure(fig, ax)
            assert not (tmp_path / "huac_scatterplot.png").exists()
        finally:
            plt.close(fig)

    def test_missing_fine_max_upstream_area(self):
        table = _huac_table().drop(columns=["fine_max_upstream_area"])
        with pytest.raises(ValueError, match="fine_max_upstream_area"):
            huac_scatterplot(table, output=None)

    def test_default_axis_labels(self):
        fig, ax = huac_scatterplot(_huac_table(), output=None)
        try:
            assert ax.get_xlabel() == r"Reference upstream area, km$^2$"
            assert ax.get_ylabel() == r"Upscaled upstream area, km$^2$"
        finally:
            plt.close(fig)

    def test_custom_axis_labels(self):
        fig, ax = huac_scatterplot(
            _huac_table(),
            xlabel="Custom x",
            ylabel="Custom y",
            output=None,
        )
        try:
            assert ax.get_xlabel() == "Custom x"
            assert ax.get_ylabel() == "Custom y"
        finally:
            plt.close(fig)

    def test_fit_line_is_dashed_and_thin(self):
        fig, ax = huac_scatterplot(_huac_table(), stats=True, output=None)
        try:
            lines = ax.get_lines()
            assert len(lines) == 1
            line = lines[0]
            assert line.get_linestyle() == "--"
            assert line.get_linewidth() == 0.6
        finally:
            plt.close(fig)


class TestWindrosePlot:
    def test_saves_png(self, tmp_path):
        path = tmp_path / "windrose_plot.png"
        fig, ax = windrose_plot(_windrose(), output=path)
        try:
            _assert_figure(fig, ax)
            _assert_png(path)
            assert ax.name == "polar"
        finally:
            plt.close(fig)

    def test_output_none_skips_default_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fig, ax = windrose_plot(_windrose(), output=None)
        try:
            _assert_figure(fig, ax)
            assert not (tmp_path / "windrose_plot.png").exists()
        finally:
            plt.close(fig)

    @pytest.mark.parametrize(
        "windrose",
        [np.zeros(7), np.full(8, 0.2)],
    )
    def test_rejects_bad_distribution(self, windrose):
        with pytest.raises(ValueError):
            windrose_plot(windrose, output=None)


class TestUpscaledMap:
    def test_saves_png(self, tmp_path):
        path = tmp_path / "upscaled_map.png"
        fig, ax = upscaled_map(
            _watershed(),
            basemap_provider=None,
            output=path,
        )
        try:
            _assert_figure(fig, ax)
            _assert_png(path)
        finally:
            plt.close(fig)

    def test_output_none_skips_default_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fig, ax = upscaled_map(
            _watershed(),
            basemap_provider=None,
            output=None,
        )
        try:
            _assert_figure(fig, ax)
            assert not (tmp_path / "upscaled_map.png").exists()
        finally:
            plt.close(fig)

    def test_shapely_geometry_without_crs(self):
        with pytest.raises(ValueError):
            upscaled_map(box(0, 0, 1, 1), basemap_provider=None, output=None)

    def test_un_scaling_requires_flowacc(self):
        with pytest.raises(ValueError, match="flowacc"):
            upscaled_map(
                _watershed(),
                upscaled_network=_network(),
                un_scaling=True,
                basemap_provider=None,
                output=None,
            )

    def test_map_has_a_full_border(self):
        fig, ax = upscaled_map(
            _watershed(),
            basemap_provider=None,
            output=None,
        )
        try:
            assert ax.get_xticks().size == 0
            assert ax.get_yticks().size == 0
            assert not any(line.get_visible() for line in ax.get_xgridlines())
            assert not any(line.get_visible() for line in ax.get_ygridlines())
            for name in ("top", "right", "bottom", "left"):
                spine = ax.spines[name]
                assert spine.get_visible()
                assert spine.get_linewidth() == 0.8
                assert tuple(float(c) for c in spine.get_edgecolor()[:3]) == (0.0, 0.0, 0.0)
        finally:
            plt.close(fig)

    def test_osm_basemap_is_called_offline(self, monkeypatch):
        import contextily

        seen = {}

        def _fake_add_basemap(ax, source=None, crs=None, reset_extent=False, **kwargs):
            seen["source"] = source
            seen["crs"] = crs

        monkeypatch.setattr(contextily, "add_basemap", _fake_add_basemap)
        fig, ax = upscaled_map(
            _watershed(),
            basemap_provider="OSM",
            output=None,
        )
        try:
            _assert_figure(fig, ax)
            assert seen["source"] is contextily.providers.OpenStreetMap.Mapnik
            assert seen["crs"] == "EPSG:3857"
        finally:
            plt.close(fig)
