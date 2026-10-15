#!/usr/bin/env python3
"""Generate the demonstration figures used in the documentation.

The scenes are SYNTHETIC. They exist to show what pygeofetch's own functions
produce; every figure is labelled as synthetic on the page that uses it.
Run from the repository root with pygeofetch installed:

    python docs/tools/gen_figures.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from pygeofetch.processing.indices import SpectralIndices
from pygeofetch.processing.sar import SARProcessor

OUT = Path(__file__).resolve().parent.parent / "_static" / "img"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
N = 360
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "figure.dpi": 140})


def write(path, arr, nodata=-9999.0):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", crs="EPSG:32630", transform=from_origin(600000, 700000, 20, 20),
                nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def read(path):
    with rasterio.open(path) as s:
        return s.read(1)


def landscape():
    """Return (green, red, nir) reflectances for a river + fields + forest + town."""
    yy, xx = np.mgrid[0:N, 0:N]
    land = np.zeros((N, N), int)                         # 0 bare/field
    z = lambda sg: (lambda a: (a - a.mean()) / a.std())(gaussian_filter(RNG.normal(size=(N, N)), sg))
    forest = z(45) > 0.25
    land[forest] = 1
    river = np.abs(yy - (0.55 * N + 40 * np.sin(xx / 55.0))) < 9
    land[river] = 2
    land[(yy > 40) & (yy < 110) & (xx > 220) & (xx < 330)] = 3   # town
    fields = z(14) > 0.15
    land[(land == 0) & fields] = 4
    refl = {  # green, red, nir
        0: (0.11, 0.17, 0.26), 1: (0.05, 0.04, 0.42), 2: (0.07, 0.05, 0.03),
        3: (0.16, 0.19, 0.22), 4: (0.09, 0.07, 0.36)}
    bands = [np.zeros((N, N)) for _ in range(3)]
    for k, v in refl.items():
        for b in range(3):
            bands[b][land == k] = v[b]
    for b in range(3):
        bands[b] = np.clip(bands[b] + gaussian_filter(RNG.normal(0, 0.02, (N, N)), 1.2), 0.001, 1)
    return bands


def fig_indices(tmp):
    g, r, n = landscape()
    paths = {}
    for k, a in (("green", g), ("red", r), ("nir", n)):
        paths[k] = tmp / f"{k}.tif"
        write(paths[k], a)
    si = SpectralIndices()
    si.ndvi(red=paths["red"], nir=paths["nir"], output=str(tmp / "ndvi.tif"))
    si.ndwi(green=paths["green"], nir=paths["nir"], output=str(tmp / "ndwi.tif"))
    ndvi, ndwi = read(tmp / "ndvi.tif"), read(tmp / "ndwi.tif")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ax[0].imshow(np.clip(np.dstack([n, r, g]) / 0.45, 0, 1))
    ax[0].set_title("False-colour input (NIR, Red, Green)")
    im1 = ax[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9); ax[1].set_title("NDVI")
    im2 = ax[2].imshow(ndwi, cmap="BrBG", vmin=-0.8, vmax=0.8); ax[2].set_title("NDWI")
    fig.colorbar(im1, ax=ax[1], fraction=0.046); fig.colorbar(im2, ax=ax[2], fraction=0.046)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Spectral indices computed with client.indices (synthetic scene)", y=1.02, fontsize=10, color="#555")
    fig.tight_layout(); fig.savefig(OUT / "demo_indices.png", bbox_inches="tight"); plt.close(fig)


def sar_scene():
    yy, xx = np.mgrid[0:N, 0:N]
    base = -9 + gaussian_filter(RNG.normal(0, 1, (N, N)), 25) * 4          # dB land
    flood = (np.hypot(xx - 0.36 * N, yy - 0.62 * N) < 70) | ((np.abs(yy - 0.3 * N) < 12) & (xx < 0.6 * N))
    flood = gaussian_filter(flood.astype(float), 3) > 0.5
    db = np.where(flood, -21 + RNG.normal(0, 0.6, (N, N)), base)
    lin = 10 ** (db / 10)
    speckle = RNG.gamma(shape=4, scale=1 / 4, size=(N, N))                # 4-look speckle
    return lin * speckle, flood


def fig_sar(tmp):
    noisy, truth = sar_scene()
    write(tmp / "sar_lin.tif", noisy)
    sp = SARProcessor()
    sp.despeckle(input=tmp / "sar_lin.tif", filter="lee", window=7, output=str(tmp / "lee.tif"))
    lee = read(tmp / "lee.tif")
    to_db = lambda a: 10 * np.log10(np.clip(a, 1e-6, None))
    noisy_db, lee_db = to_db(noisy), to_db(lee)
    write(tmp / "lee_db.tif", lee_db)
    sp.flood_map(input=tmp / "lee_db.tif", threshold=-15.0, output=str(tmp / "flood.tif"))
    flood = read(tmp / "flood.tif")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    for a, d, t in ((ax[0], noisy_db, "Raw backscatter (dB)"), (ax[1], lee_db, "After Lee filter, 7 x 7")):
        im = a.imshow(d, cmap="gray", vmin=-26, vmax=-2); a.set_title(t)
    fig.colorbar(im, ax=ax[:2], fraction=0.025, label="dB")
    ax[2].imshow(lee_db, cmap="gray", vmin=-26, vmax=-2)
    ax[2].imshow(np.ma.masked_where(flood != 1, flood), cmap="winter", alpha=0.75)
    ax[2].set_title("Flood mask at -15 dB")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("client.sar: despeckle then flood map (synthetic scene)", y=1.02, fontsize=10, color="#555")
    fig.savefig(OUT / "demo_sar.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        fig_indices(t)
        fig_sar(t)
    print("figures written to", OUT)
