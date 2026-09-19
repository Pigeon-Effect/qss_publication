"""Draw the semantic landscape of AI subdisciplines (manuscript Figure 1).

Every micro-cluster is drawn as its density outline, filled in the colour of the
macro-domain it belongs to and labelled with its three-digit code, so the
hierarchy can be read off the map: the first digit is the domain, the colour
repeats it, and the remaining digits locate the field and the research front.

By default the script draws the layout deposited in ``resources/``, which is the
one published as Figure 1. Point ``--layout`` at the file written by
``summarize_clusters.py`` to draw a layout computed from a new projection.

Note on reading the map. Position and adjacency carry meaning: clusters that sit
next to each other share vocabulary. Outline size is semantic spread, not
publication volume. The rotation of a UMAP layout is arbitrary, so absolute
directions carry no meaning; only relative arrangement does.

Input  - a landscape layout in GeoJSON (``resources/semantic_landscape_layout.geojson``
         by default), optionally the projected points behind it
Output - ``output/semantic_landscape.svg`` (and ``.png`` with ``--png``)
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/06_visualization/semantic_landscape/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PUBLISHED_LAYOUT = HERE / "resources" / "semantic_landscape_layout.geojson"
OUTPUT_DIR = HERE.parent / "output"
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

# Macro-domain colours, as used throughout the manuscript.
MACRO_COLORS = {
    0: "#7b2cbf",  # Computer Science
    1: "#3a86ff",  # Health Science
    2: "#2dd4bf",  # Social Science
    3: "#ffd500",  # Natural Science
    4: "#fb5607",  # Engineering
}

FILL_ALPHA = 0.8
EDGE_WIDTH = 0.75
EDGE_COLOR = "#000000"
LABEL_SIZE = 6.5
LABEL_TEXT_COLOR = "#111111"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable and re-fontable


def load_layout(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Layout not found: {path}\n"
            "Pass --layout, or run summarize_clusters.py to compute one."
        )
    with open(path, encoding="utf-8") as handle:
        collection = json.load(handle)
    features = collection.get("features", [])
    if not features:
        raise SystemExit(f"No features in {path}")
    return features


def draw(features: list[dict], out_svg: Path, out_png: Path | None,
         points_csv: Path | None, width_in: float, labels: bool) -> None:
    rings = {f["properties"]["code"]: np.asarray(f["geometry"]["coordinates"][0], dtype=float)
             for f in features}
    all_xy = np.vstack(list(rings.values()))
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    span_x, span_y = x_max - x_min, y_max - y_min

    height_in = width_in * (span_y / span_x)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_aspect("equal")
    ax.axis("off")

    if points_csv is not None:
        import pandas as pd
        frame = pd.read_csv(points_csv)
        colours = [MACRO_COLORS.get(int(h), "#999999") for h in frame["h1_cluster"]]
        ax.scatter(frame["x"], frame["y"], s=0.6, c=colours, alpha=0.18, linewidths=0)

    # Largest first, so a small cluster drawn on top of a bigger one stays visible.
    def area(ring: np.ndarray) -> float:
        x, y = ring[:, 0], ring[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    for feature in sorted(features, key=lambda f: -area(rings[f["properties"]["code"]])):
        props = feature["properties"]
        ring = rings[props["code"]]
        ax.add_patch(PolygonPatch(
            ring, closed=True,
            facecolor=MACRO_COLORS.get(int(props["h1"]), "#999999"),
            edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, alpha=FILL_ALPHA, zorder=2,
        ))

    if labels:
        for feature in features:
            props = feature["properties"]
            ring = rings[props["code"]]
            x = props.get("label_x", float(ring[:, 0].mean()))
            y = props.get("label_y", float(ring[:, 1].mean()))
            ax.text(
                x, y, props["code"], ha="center", va="center",
                fontsize=LABEL_SIZE, color=LABEL_TEXT_COLOR, zorder=3,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor=EDGE_COLOR, linewidth=0.5),
            )

    margin_x, margin_y = span_x * 0.02, span_y * 0.02
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    print(f"Written: {out_svg}")
    if out_png is not None:
        fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Written: {out_png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", type=Path, default=PUBLISHED_LAYOUT,
                        help="landscape layout in GeoJSON (default: the published one)")
    parser.add_argument("--points", type=Path, default=None,
                        help="also draw the projected scatter behind the outlines")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "semantic_landscape.svg")
    parser.add_argument("--png", action="store_true", help="also write a PNG next to the SVG")
    parser.add_argument("--width", type=float, default=11.0, help="figure width in inches")
    parser.add_argument("--no-labels", action="store_true", help="draw the outlines without codes")
    args = parser.parse_args()

    features = load_layout(args.layout)
    print(f"Drawing {len(features)} micro-cluster outlines from {args.layout.name}")
    draw(
        features,
        out_svg=args.output,
        out_png=args.output.with_suffix(".png") if args.png else None,
        points_csv=args.points,
        width_in=args.width,
        labels=not args.no_labels,
    )


if __name__ == "__main__":
    main()
