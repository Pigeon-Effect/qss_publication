"""Summarise each micro-cluster's point cloud as a density outline (manuscript section 4.1).

The UMAP projection written by ``project_sample.py`` is a scatter of some
hundred thousand points. Drawn directly it is an unreadable cloud, and the
overlap between neighbouring research fronts hides exactly the structure the
figure is meant to show. This step replaces each micro-cluster's scatter with a
single outline:

1. A Gaussian kernel density estimate is computed over the cluster's own points,
   evaluated on a regular grid spanning that cluster's extent. The kernel
   bandwidth follows Scott's rule and is scaled by the cluster's covariance, so
   an elongated cluster gets an elongated kernel.
2. The density level that encloses ``--coverage`` of the cluster's points
   (80 % by default) is read off the density values at the points themselves:
   the 20th percentile of those values is the level below which the sparsest
   20 % of points sit.
3. The contour at that level is traced, and the largest closed ring is kept as
   the cluster's outline.

Because the outline follows a density level rather than the outermost points,
its size reflects how far a cluster spreads through the embedding space - its
semantic spread - and not how many publications it holds. A tight, populous
cluster is drawn small; a small but diffuse one is drawn large. Outliers do not
inflate it, which is what makes the map readable.

Input  - ``data/interim/semantic_landscape/points.csv.gz`` from ``project_sample.py``
Output - ``data/interim/semantic_landscape/semantic_landscape_layout.geojson``,
         one polygon per micro-cluster, ready for ``render_landscape.py``
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/06_visualization/semantic_landscape/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
LANDSCAPE_DIR = INTERIM_DIR / "semantic_landscape"
TAXONOMY_CSV = PROJECT_ROOT / "code" / "05_impact_analysis" / "taxonomy.csv"


def gaussian_density(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian KDE fitted on `points` at every row of `grid`.

    Bandwidth follows Scott's rule for two dimensions (n ** (-1/6)) applied to
    the sample covariance, which is what `scipy.stats.gaussian_kde` does; it is
    reimplemented here so the pipeline needs nothing beyond NumPy.
    """
    n = len(points)
    covariance = np.cov(points.T)
    covariance = np.atleast_2d(covariance) * n ** (-1.0 / 3.0)  # (n ** (-1/6)) ** 2

    # Guard against a singular covariance (e.g. perfectly collinear points).
    spread = float(np.mean(np.diag(covariance))) or 1.0
    covariance = covariance + np.eye(2) * spread * 1e-6

    inverse = np.linalg.inv(covariance)
    scale = 1.0 / (n * 2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))

    density = np.zeros(len(grid))
    for start in range(0, n, 512):  # chunked to keep memory bounded
        chunk = points[start: start + 512]
        delta = grid[:, None, :] - chunk[None, :, :]
        mahalanobis = np.einsum("gpi,ij,gpj->gp", delta, inverse, delta)
        density += np.exp(-0.5 * mahalanobis).sum(axis=1)
    return density * scale


def polygon_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_centroid(ring: np.ndarray) -> tuple[float, float]:
    x, y = ring[:, 0], ring[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = float(((x + np.roll(x, -1)) * cross).sum() / (6.0 * area))
    cy = float(((y + np.roll(y, -1)) * cross).sum() / (6.0 * area))
    return cx, cy


def outline_for_cluster(points: np.ndarray, coverage: float, resolution: int,
                        padding: float, rng: np.random.Generator,
                        max_points: int) -> Optional[np.ndarray]:
    """Return the density outline of one cluster, or None if it is too small."""
    if len(points) < 20:
        return None
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]

    (x_min, y_min), (x_max, y_max) = points.min(axis=0), points.max(axis=0)
    pad_x = (x_max - x_min) * padding or 1.0
    pad_y = (y_max - y_min) * padding or 1.0
    xs = np.linspace(x_min - pad_x, x_max + pad_x, resolution)
    ys = np.linspace(y_min - pad_y, y_max + pad_y, resolution)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    grid = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    density_grid = gaussian_density(points, grid).reshape(mesh_x.shape)
    density_points = gaussian_density(points, points)
    level = float(np.quantile(density_points, 1.0 - coverage))

    figure = plt.figure()
    try:
        contour = plt.contour(mesh_x, mesh_y, density_grid, levels=[level])
        rings = [np.asarray(seg) for seg in contour.allsegs[0] if len(seg) >= 3]
    finally:
        plt.close(figure)

    if not rings:
        return None
    ring = max(rings, key=polygon_area)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    return ring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=Path, default=LANDSCAPE_DIR / "points.csv.gz")
    parser.add_argument("--output", type=Path, default=LANDSCAPE_DIR / "semantic_landscape_layout.geojson")
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_CSV)
    parser.add_argument("--coverage", type=float, default=0.8,
                        help="share of a cluster's points the outline encloses")
    parser.add_argument("--resolution", type=int, default=100, help="density grid cells per axis")
    parser.add_argument("--padding", type=float, default=0.05,
                        help="grid margin around a cluster's extent, as a fraction of it")
    parser.add_argument("--max-points", type=int, default=5000,
                        help="points per cluster used for the density estimate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.points.exists():
        raise SystemExit(
            f"Projected points not found: {args.points}\n"
            "Run project_sample.py first, or pass --points."
        )

    frame = pd.read_csv(args.points)
    for column in ("x", "y", "h1_cluster", "h2_cluster", "h3_cluster"):
        if column not in frame.columns:
            raise SystemExit(f"Column '{column}' missing from {args.points}")

    frame = frame.dropna(subset=["x", "y", "h1_cluster", "h2_cluster", "h3_cluster"])
    frame["code"] = (frame["h1_cluster"].astype(int).astype(str)
                     + frame["h2_cluster"].astype(int).astype(str)
                     + frame["h3_cluster"].astype(int).astype(str))

    taxonomy = {}
    if args.taxonomy.exists():
        with open(args.taxonomy, encoding="utf-8-sig") as handle:
            taxonomy = {row["code"]: row for row in csv.DictReader(handle)}

    rng = np.random.default_rng(args.seed)
    features = []
    skipped = []
    for code, group in sorted(frame.groupby("code"), key=lambda kv: kv[0]):
        points = group[["x", "y"]].to_numpy(dtype=float)
        ring = outline_for_cluster(points, args.coverage, args.resolution,
                                   args.padding, rng, args.max_points)
        if ring is None:
            skipped.append(code)
            continue
        label_x, label_y = polygon_centroid(ring)
        names = taxonomy.get(code, {})
        features.append({
            "type": "Feature",
            "properties": {
                "code": code,
                "h1": int(code[0]), "h2": int(code[1]), "h3": int(code[2]),
                "macro": names.get("macro", ""), "meso": names.get("meso", ""),
                "micro": names.get("micro", ""),
                "n_points": int(len(points)),
                "coverage": args.coverage,
                "label_x": round(label_x, 2), "label_y": round(label_y, 2),
            },
            "geometry": {"type": "Polygon",
                         "coordinates": [[[round(float(x), 3), round(float(y), 3)] for x, y in ring]]},
        })
        print(f"{code}: {len(points):>6,} points -> outline with {len(ring)} vertices")

    collection = {
        "type": "FeatureCollection",
        "name": "semantic_landscape_layout",
        "description": (f"Density outlines enclosing {args.coverage:.0%} of each micro-cluster's "
                        "points in the SPECTER/UMAP embedding space. Coordinates are UMAP units; "
                        "only relative position, size and shape carry meaning."),
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(collection, handle, indent=1)
        handle.write("\n")

    print(f"\nOutlines written for {len(features)} micro-clusters: {args.output}")
    if skipped:
        print(f"Clusters with too few points to summarise: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
