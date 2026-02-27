"""Base class for flow-direction upscaling algorithms."""

import numpy as np
import rasterio
from rasterio.transform import Affine


class BaseUpscaler:
    """Common infrastructure for flow-direction upscalers.

    Subclasses must override ``upscale(k, ...)``.  The typical workflow is::

        upscaler = SomeUpscaler()
        upscaler.load_flowacc("flowacc.tif")
        cells = upscaler.upscale(k=4)
        upscaler.save("output.tif")
    """

    DIR_NODATA = np.uint8(255)

    def __init__(self):
        self._flowacc_raw = None
        self._flowacc_nodata = None
        self._flowacc_padded = None
        self._orig_shape = None
        self._padded_k = None
        self._profile = None
        self.cells_ = None
        self.k_ = None

    # ------------------------------------------------------------------
    # Raster I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _read_raster(path):
        """Read a single-band raster via rasterio.

        Returns (array, profile, nodata).
        """
        with rasterio.open(path) as src:
            array = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata
        return array, profile, nodata

    def load_flowacc(self, path):
        """Load a flow-accumulation raster (single band).

        Stores the raw array, rasterio profile, and nodata value.
        Padding and masking are deferred to ``upscale()``.
        """
        array, profile, nodata = self._read_raster(path)
        self._flowacc_raw = array
        self._flowacc_nodata = nodata
        self._profile = profile

    # ------------------------------------------------------------------
    # Nodata / padding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nodata_mask(array, nodata):
        """Boolean mask where *True* means nodata."""
        if nodata is None:
            return np.zeros(array.shape, dtype=bool)
        if np.issubdtype(type(nodata), np.floating) and np.isnan(nodata):
            return np.isnan(array)
        return array == nodata

    @staticmethod
    def _pad_to_multiple(array, k, pad_value=0):
        """Pad *array* so both dimensions are divisible by *k*."""
        nrows, ncols = array.shape
        add_rows = (k - nrows % k) % k
        add_cols = (k - ncols % k) % k
        if add_rows or add_cols:
            array = np.pad(
                array,
                pad_width=((0, add_rows), (0, add_cols)),
                mode="constant",
                constant_values=pad_value,
            )
        return array

    @staticmethod
    def _repad(padded, orig_shape, k, pad_value=0):
        """Crop *padded* back to *orig_shape* then re-pad it to a multiple of *k*."""
        r, c = orig_shape
        return BaseUpscaler._pad_to_multiple(padded[:r, :c], k, pad_value)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, path):
        """Write ``self.cells_`` to a uint8 GeoTIFF with nodata = 255.

        The spatial transform is adjusted by the scaling factor stored
        during the last ``upscale()`` call.
        """
        if self.cells_ is None or self.k_ is None:
            raise RuntimeError(
                "No upscaling result to save. Call upscale() first."
            )
        if self._profile is None:
            raise RuntimeError(
                "No raster profile available. Call load_flowacc() first."
            )

        src_transform = self._profile["transform"]
        out_transform = Affine(
            src_transform.a * self.k_,
            src_transform.b,
            src_transform.c,
            src_transform.d,
            src_transform.e * self.k_,
            src_transform.f,
        )

        out_profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": self.cells_.shape[1],
            "height": self.cells_.shape[0],
            "count": 1,
            "crs": self._profile.get("crs"),
            "transform": out_transform,
            "nodata": int(self.DIR_NODATA),
        }

        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(self.cells_, 1)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def upscale(self, k, **kwargs):
        """Run the upscaling algorithm.

        Must store results in ``self.cells_`` (uint8 ndarray) and
        ``self.k_`` (int), then return ``self.cells_.copy()``.

        Subclasses must override this method.
        """
        raise NotImplementedError
