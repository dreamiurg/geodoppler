"""Ground elevation from Copernicus GLO-90 (AWS open data, 1x1 degree COG tiles)."""

import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif"


def _name(lat_i: int, lon_i: int) -> str:
    ns = f"N{lat_i:02d}" if lat_i >= 0 else f"S{-lat_i:02d}"
    ew = f"E{lon_i:03d}" if lon_i >= 0 else f"W{-lon_i:03d}"
    return f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"


class Elevation:
    """Callable (lat[], lon[]) -> metres. Ocean tiles (absent from the bucket) are 0."""

    def __init__(self, cache: Path):
        self.cache = cache / "dem"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.tiles: dict[tuple[int, int], np.ndarray | None] = {}

    def _tile(self, key):
        if key not in self.tiles:
            n = _name(*key)
            p = self.cache / f"{n}.tif"
            if not p.exists() and not p.with_suffix(".none").exists():
                tmp = p.with_suffix(".part")
                try:
                    urllib.request.urlretrieve(URL.format(n=n), tmp)
                    tmp.rename(p)
                except urllib.error.HTTPError as e:
                    if e.code not in (403, 404):  # missing tile = sea
                        raise
                    p.with_suffix(".none").touch()
            if p.exists():
                import rasterio
                with rasterio.open(p) as r:
                    self.tiles[key] = np.asarray(r.read(1), dtype=np.float32)
            else:
                self.tiles[key] = None
        return self.tiles[key]

    def __call__(self, lat, lon):
        lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
        out = np.zeros(lat.shape, np.float32)
        li, lo = np.floor(lat).astype(int), np.floor(lon).astype(int)
        for key in set(zip(li.tolist(), lo.tolist())):
            a = self._tile(key)
            if a is None:
                continue
            m = (li == key[0]) & (lo == key[1])
            h, w = a.shape  # row 0 = north edge; width varies with latitude
            r = np.clip(((key[0] + 1 - lat[m]) * h).astype(int), 0, h - 1)
            c = np.clip(((lon[m] - key[1]) * w).astype(int), 0, w - 1)
            out[m] = np.maximum(a[r, c], 0)
        return out

    def __getstate__(self):  # workers reload tiles from disk rather than pickling arrays
        return {"cache": self.cache, "tiles": {}}

    def __setstate__(self, s):
        self.cache, self.tiles = s["cache"], s["tiles"]


if __name__ == "__main__":
    e = Elevation(Path.home() / ".cache" / "pinpoint")
    print(e([40.87, 39.7392, 36.0544], [-73.52, -104.9903, -112.1401]))  # Oyster Bay, Denver, Grand Canyon
    assert math.isfinite(float(e([0.0], [-30.0])[0]))
