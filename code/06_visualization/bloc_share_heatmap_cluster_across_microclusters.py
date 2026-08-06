#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Five-column H1 layout — hierarchical treemap-style boxes, wider H1/H2 frames, centered meso labels.

New in this revision:
  • All H1 columns are forced to have the same total height as the tallest column by
    distributing extra vertical space *uniformly* across available "slots":
      - after the TOTAL heatmap (1 slot),
      - between rows inside each meso (n_rows - 1 slots per meso),
      - between meso boxes (G - 1 slots).
    The per-column stretch is computed as:
      stretch_px = max(0, (target_content_h - baseline_content_h) / slots)
    and applied consistently both in the geometry precompute and the draw pass.

Carried over:
  • Fonts scaled by ~+50% (SCALE=1.5).
  • TOTAL row at the top.
  • Heatmap height equals colorbar height.
  • Tight inner layout; labels centered.

Updates:
  • H2 (Meso) titles are larger, with increased top padding and bottom spacing.

CSV columns required:
  ['macro_id','macro','meso','micro','code','total_citations',
   'cn_cits','us_cits','eu27_cits','row_cits',
   'cn_share','us_share','eu27_share','row_share']
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch

# ── Figure + geometry ─────────────────────────────────────────────────────
plt.rcParams["font.family"] = ("Times New Roman")
plt.rcParams['svg.fonttype'] = 'none'  # <--- ADD THIS LINE

# ── Paths / Config ────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/06_visualization/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

# Per-cluster, per-bloc fractional citation shares. Produced by stage 05
# (impact analysis), which is not yet in this repository - see its README.
IN_CSV = Path(os.environ.get("QSS_CITSHARE_CSV", INTERIM_DIR / "citshare_h3x4entities.csv"))
OUT_SVG = Path(__file__).resolve().parent / "output" / "bloc_citation_shares_across_subdisciplines.svg"

# Global font scale (~+50%)
SCALE = 1.5

MAIN_FONT = "Times New Roman"
LABEL_FSIZE = int(round(8 * SCALE))  # micro labels + % text
XTICK_FSIZE = int(round(8 * SCALE))  # entity names header
ANNOT_FSIZE = int(round(8 * SCALE))  # numbers inside tiles
TITLE_FSIZE = int(round(10 * SCALE))  # macro titles above column
CBAR_FSIZE = int(round(8 * SCALE))  # colorbar ticks

# --- UPDATED H2 SETTINGS ---
MESO_FSIZE = int(round(10 * SCALE))  # Increased from 8 -> 10
# ---------------------------

ENTITIES = ["China", "USA", "EU-27", "RoW"]
VALUE_COLS = ["cn_share", "us_share", "eu27_share", "row_share"]

# Figure / layout
FIG_W_IN = 22.0
TOP_FRAC = 0.95
BOT_FRAC = 0.12
LEFT_FRAC = 0.00
RIGHT_FRAC = 0.00

# Column spacing (increased to allow wider boxes)
COL_GAP_PX = 60  # px between H1 columns

# Vertical spacing (pixels) — scaled by DPI later
LABEL_TO_HM_GAP_PX = 3
ITEM_GAP_PX = 16
COL_TITLE_GAP_PX = 8
HEADER_GAP_BELOW_TITLE = 6
TOTAL_BLOCK_TOP_PAD_PX = 8

# Extra spacing below titles
TITLE_BOTTOM_GAP_PX = 6

# --- UPDATED SPACING BELOW ---
MESO_TITLE_BOTTOM_GAP_PX = 12  # Increased from 4 -> 12 (Space below H2 title)
# -----------------------------

# Treemap-style boxes
# H1 column box (now using macro colors per column)
COL_BOX_EDGE_COLOR = "black"
COL_BOX_FACE_ALPHA = 1.0
COL_BOX_EDGE_ALPHA = 1.0
COL_BOX_PAD_PX = 10
COL_BOX_OUTSET_PX = 24
COL_BOX_RADIUS_PX = 14  # radius in *pixels* for H1
COL_BOX_EXTRA_BOTTOM_PX = 10  # extra bottom margin

# H2 meso boxes (off-white), wider than column but less than H1
MESO_FACE_COLOR = "#ebeae5"  # updated from white to off-white
MESO_EDGE_COLOR = "black"
MESO_FACE_ALPHA = 1.0
MESO_EDGE_ALPHA = 1.0

# --- UPDATED PADDING BELOW ---
MESO_PAD_PX = 14  # Increased from 8 -> 14 (Top padding inside H2 box)
# -----------------------------

MESO_BOX_GAP_PX = 12
MESO_BOX_OUTSET_PX = 8
MESO_BOX_RADIUS_PX = 10  # radius in *pixels* for H2

# Mini-heatmap geometry
HEATMAP_ASPECT = 5.0  # width : height

# Per-cell rounded box settings for the mini heatmaps
CELL_GAP_PX = 2  # inner padding between cells and around edges
CELL_RADIUS_PX = 4  # rounded-corner radius for each cell (pixels)

# Colormap range
V_MIN, V_MAX = 0.0, 0.50

# H1 column box colors (per macro_id) – cluster colors
MACRO_COLORS = {
    0: "#9556cb",  # first column
    1: "#619eff",  # second
    2: "#57dccb",  # third
    3: "#ffdd33",  # fourth
    4: "#fb7738",  # fifth
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _code_int(s) -> int:
    s = str(s)
    return int(s) if len(s) == 3 and s.isdigit() else 10_000


def text_color_for_value(v: float, cmap) -> str:
    r, g, b, _ = cmap(np.clip((v - V_MIN) / (V_MAX - V_MIN), 0, 1))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


def px_to_pt(px: float, dpi: float) -> float:
    return px * 72.0 / dpi


def draw_heatmap_row(ax, values: np.ndarray, hm_w_px: float, hm_h_px: float,
                     cmap, norm):
    """
    Draw a 1×N heatmap as N rounded rectangles with black borders and spacing
    inside an axis whose data coordinates are in pixels (0..hm_w_px, 0..hm_h_px).
    """
    n = len(values)
    ax.set_xlim(0, hm_w_px)
    ax.set_ylim(0, hm_h_px)
    ax.axis("off")

    if n <= 0 or hm_w_px <= 0 or hm_h_px <= 0:
        return

    cell_w = hm_w_px / n
    y0 = CELL_GAP_PX
    h = hm_h_px - 2 * CELL_GAP_PX
    if h <= 0:
        return

    for i, val in enumerate(values):
        x0 = i * cell_w + CELL_GAP_PX
        w = cell_w - 2 * CELL_GAP_PX
        if w <= 0:
            continue

        if np.isfinite(val):
            face = cmap(norm(val))
        else:
            face = (1, 1, 1, 1)  # white for NaN

        patch = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={CELL_RADIUS_PX}",
            transform=ax.transData,
            facecolor=face,
            edgecolor="black",
            linewidth=0.6,
            antialiased=True,
            zorder=0
        )
        ax.add_patch(patch)

        if np.isfinite(val):
            ax.text(
                x0 + w / 2, y0 + h / 2,
                f"{val * 100:.1f}",
                ha="center", va="center",
                fontsize=ANNOT_FSIZE,
                color=text_color_for_value(val, cmap),
                zorder=1
            )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

    required = [
                   "macro_id", "macro", "meso", "micro", "code", "total_citations",
                   "cn_cits", "us_cits", "eu27_cits", "row_cits"
               ] + VALUE_COLS
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    if not np.issubdtype(df["macro_id"].dtype, np.integer):
        df["macro_id"] = pd.to_numeric(df["macro_id"], errors="coerce").fillna(-1).astype(int)

    # Sort by macro, then numeric code
    df = df.sort_values(by=["macro_id", "code"], key=lambda s: s.map(_code_int)).reset_index(drop=True)

    # Corpus-level % for labels
    global_total_cits = float(df["total_citations"].sum())
    if global_total_cits <= 0:
        raise ValueError("Global total_citations is zero or missing; cannot compute corpus-level percentages.")
    df["_global_pct_cits"] = (df["total_citations"] / global_total_cits * 100.0)

    # Build per-macro structures (with meso grouping)
    macro_ids: List[int] = sorted([m for m in df["macro_id"].unique() if m >= 0]) or [0]
    if macro_ids == [0]:
        df["macro_id"] = 0

    panels: List[Dict[str, Any]] = []
    for mid in macro_ids:
        sub = df[df["macro_id"] == mid].copy()

        macro_title = (sub["macro"].dropna().astype(str).unique() or [f"Macro {mid}"])[0]

        # Macro totals
        macro_total_cits = float(sub["total_citations"].sum())
        total_pct_str = f"{(macro_total_cits / global_total_cits * 100.0):.2f}%"

        agg_cn = float(sub["cn_cits"].sum())
        agg_us = float(sub["us_cits"].sum())
        agg_eu = float(sub["eu27_cits"].sum())
        agg_row = float(sub["row_cits"].sum())
        denom = macro_total_cits if macro_total_cits > 0 else np.nan
        macro_row = np.array([
            (agg_cn / denom) if denom else np.nan,
            (agg_us / denom) if denom else np.nan,
            (agg_eu / denom) if denom else np.nan,
            (agg_row / denom) if denom else np.nan
        ], dtype=float)

        # Sequences for grouping
        meso_names = sub["meso"].fillna("Unknown").astype(str).tolist()
        micro_names = sub["micro"].fillna(sub["code"]).astype(str).tolist()
        pct_strs = [f"{p:.2f}%" for p in sub["_global_pct_cits"].to_numpy(dtype=float)]
        mats = sub[VALUE_COLS].to_numpy(dtype=float)

        # Build ordered meso groups (contiguous blocks)
        groups = []
        start = 0
        for i in range(1, len(meso_names) + 1):
            if i == len(meso_names) or meso_names[i] != meso_names[start]:
                groups.append({"meso": meso_names[start], "rows": list(range(start, i))})
                start = i

        panels.append({
            "macro_id": mid,
            "title": macro_title,
            "total_pct": total_pct_str,
            "macro_row": macro_row,
            "meso_groups": groups,
            "micro_labels": micro_names,
            "pct_labels": pct_strs,
            "matrices": mats,
        })

    # ── Figure + geometry ─────────────────────────────────────────────────────
    plt.rcParams["font.family"] = MAIN_FONT

    # Custom grey colormap: black -> mid-grey -> white
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_grey", ["#FFFFFF", "#808080", "#000000"]
    )
    norm = mpl.colors.Normalize(vmin=V_MIN, vmax=V_MAX)

    # Initial DPI for sizing
    dpi = 100

    def pt_to_px(pt: float) -> float:
        return pt * dpi / 72.0

    label_h_px = pt_to_px(LABEL_FSIZE) * 1.3
    header_h_px = pt_to_px(XTICK_FSIZE) * 1.3

    # --- CHANGED: Meso title height based on MESO_FSIZE ---
    meso_title_h_px = pt_to_px(MESO_FSIZE) * 1.3
    # ------------------------------------------------------

    # Horizontal geometry (pixels)
    fig_w_px = FIG_W_IN * dpi
    left_px = LEFT_FRAC * fig_w_px
    right_px = (1.0 - RIGHT_FRAC) * fig_w_px

    ncols = len(panels)
    inner_w_px = right_px - left_px - (ncols - 1) * COL_GAP_PX
    hm_w_px = inner_w_px / ncols
    col_w_px = hm_w_px

    # Heatmap height
    baseline_h = hm_w_px / HEATMAP_ASPECT
    hm_h_px = max(baseline_h * 0.5, 8.0)

    # Block heights (without any stretch)
    item_h_px = label_h_px + LABEL_TO_HM_GAP_PX + hm_h_px + ITEM_GAP_PX
    title_h_px = pt_to_px(TITLE_FSIZE) * 1.4 + COL_TITLE_GAP_PX
    header_block_h_px = HEADER_GAP_BELOW_TITLE + header_h_px
    total_block_h_px = TOTAL_BLOCK_TOP_PAD_PX + item_h_px  # TOTAL label + hm + default gap

    # ── First pass: baseline heights + slot counts per column ─────────────────
    col_baseline_content_heights = []
    col_total_rows = []  # total #micro rows in this macro (equals #stretch slots)
    for p in panels:
        # totals
        totals_h = total_block_h_px
        # meso part
        meso_total_h = 0
        total_rows = 0
        for gi, g in enumerate(p["meso_groups"]):
            n_rows = len(g["rows"])
            if n_rows == 0:
                continue
            total_rows += n_rows
            # baseline content of all rows in this meso (subtract one ITEM_GAP at the end)
            group_content_h = n_rows * item_h_px - ITEM_GAP_PX

            # NOTE: group_box_h accounts for increased padding and larger titles here
            group_box_h = 2 * MESO_PAD_PX + meso_title_h_px + MESO_TITLE_BOTTOM_GAP_PX + group_content_h

            meso_total_h += group_box_h
            if gi < len(p["meso_groups"]) - 1:
                meso_total_h += MESO_BOX_GAP_PX

        # Full baseline content (inside H1 box)
        content_h = title_h_px + TITLE_BOTTOM_GAP_PX + header_block_h_px + totals_h + meso_total_h
        col_baseline_content_heights.append(content_h)
        col_total_rows.append(total_rows)

    # Target content height = tallest baseline content height among columns
    target_content_h = max(col_baseline_content_heights)

    # Compute per-column stretch (px per slot); slots = total_rows
    per_col_stretch = []
    for content_h, slots in zip(col_baseline_content_heights, col_total_rows):
        if slots > 0 and target_content_h > content_h:
            stretch = (target_content_h - content_h) / float(slots)
        else:
            stretch = 0.0
        per_col_stretch.append(stretch)

    # ── Figure height (based on target content height) ────────────────────────
    max_col_h_px = target_content_h + 2 * COL_BOX_PAD_PX + COL_BOX_EXTRA_BOTTOM_PX
    fig_h_px = max(max_col_h_px / (TOP_FRAC - BOT_FRAC), 1000)
    fig_h_in = fig_h_px / dpi

    # Create figure (use actual DPI)
    fig = plt.figure(figsize=(FIG_W_IN, fig_h_in), dpi=dpi)
    dpi = fig.get_dpi()
    fig_w_px = FIG_W_IN * dpi
    fig_h_px = fig_h_in * dpi
    left_px = LEFT_FRAC * fig_w_px
    right_px = (1.0 - RIGHT_FRAC) * fig_w_px

    inner_w_px = right_px - left_px - (ncols - 1) * COL_GAP_PX
    hm_w_px = inner_w_px / ncols
    col_w_px = hm_w_px
    baseline_h = hm_w_px / HEATMAP_ASPECT
    hm_h_px = max(baseline_h * 0.5, 8.0)

    label_h_px = LABEL_FSIZE * dpi / 72.0 * 1.3
    header_h_px = XTICK_FSIZE * dpi / 72.0 * 1.3

    # --- CHANGED: Recalculate based on DPI ---
    meso_title_h_px = MESO_FSIZE * dpi / 72.0 * 1.3
    # -----------------------------------------

    item_h_px = label_h_px + LABEL_TO_HM_GAP_PX + hm_h_px + ITEM_GAP_PX
    title_h_px = TITLE_FSIZE * dpi / 72.0 * 1.4 + COL_TITLE_GAP_PX
    header_block_h_px = HEADER_GAP_BELOW_TITLE + header_h_px
    total_block_h_px = TOTAL_BLOCK_TOP_PAD_PX + item_h_px

    top_y_px = TOP_FRAC * fig_h_px
    bot_y_px = BOT_FRAC * fig_h_px

    # ── Draw columns ──────────────────────────────────────────────────────────
    for col_idx, p in enumerate(panels):
        stretch = per_col_stretch[col_idx]

        x0_col = left_px + col_idx * (col_w_px + COL_GAP_PX)
        x0_hm = x0_col  # heatmap spans full column width

        # Compute meso box heights for this column *including stretch*
        group_geom = []
        meso_total_h = 0
        total_rows = 0
        for gi, g in enumerate(p["meso_groups"]):
            n_rows = len(g["rows"])
            if n_rows == 0:
                group_geom.append(0)
                continue
            total_rows += n_rows
            # baseline group content height + stretch for (n_rows - 1) internal row gaps
            group_content_h = n_rows * item_h_px - ITEM_GAP_PX + max(0, n_rows - 1) * stretch

            # NOTE: group_box_h uses the increased padding/gaps here too
            group_box_h = 2 * MESO_PAD_PX + meso_title_h_px + MESO_TITLE_BOTTOM_GAP_PX + group_content_h

            group_geom.append(group_box_h)
            meso_total_h += group_box_h
            if gi < len(p["meso_groups"]) - 1:
                meso_total_h += MESO_BOX_GAP_PX + stretch  # add stretch to inter-meso gaps

        # Total content height for this column should equal target_content_h
        content_h = title_h_px + TITLE_BOTTOM_GAP_PX + header_block_h_px + (total_block_h_px + stretch) + meso_total_h

        # ── H1 column box FIRST (so everything else sits on top) ──────────────
        col_box_h = content_h + 2 * COL_BOX_PAD_PX + COL_BOX_EXTRA_BOTTOM_PX
        col_box_x = x0_col - COL_BOX_OUTSET_PX
        col_box_w = col_w_px + 2 * COL_BOX_OUTSET_PX

        # Axes that exactly match the H1 box size in pixels → 1 data unit = 1 px
        col_box_ax = fig.add_axes([
            col_box_x / fig_w_px,
            (top_y_px - col_box_h) / fig_h_px,
            col_box_w / fig_w_px,
            col_box_h / fig_h_px
        ])
        col_box_w_px = col_box_w
        col_box_h_px = col_box_h
        col_box_ax.set_xlim(0, col_box_w_px)
        col_box_ax.set_ylim(0, col_box_h_px)
        col_box_ax.axis("off")

        macro_color = MACRO_COLORS.get(p["macro_id"], "#666666")
        # Rounded colored box with constant pixel radius
        col_patch = FancyBboxPatch(
            (0, 0), col_box_w_px, col_box_h_px,
            boxstyle=f"round,pad=0.0,rounding_size={COL_BOX_RADIUS_PX}",
            transform=col_box_ax.transData,
            facecolor=macro_color,
            edgecolor=COL_BOX_EDGE_COLOR,
            linewidth=1.4,
            alpha=1.0,
            zorder=0,
            antialiased=True
        )
        col_box_ax.add_patch(col_patch)

        # y cursor starts inside H1 box after top padding
        y_cursor = top_y_px - COL_BOX_PAD_PX

        # Column title (INSIDE H1 box)
        title_ax = fig.add_axes([
            x0_col / fig_w_px,
            (y_cursor - title_h_px) / fig_h_px,
            col_w_px / fig_w_px,
            title_h_px / fig_h_px
        ])
        title_ax.axis("off")
        title_ax.text(0.5, 0.0, p["title"], ha="center", va="bottom", fontsize=TITLE_FSIZE)
        y_cursor -= title_h_px + TITLE_BOTTOM_GAP_PX

        # Entity header (INSIDE H1 box)
        y_cursor -= HEADER_GAP_BELOW_TITLE
        header_ax = fig.add_axes([
            x0_hm / fig_w_px,
            (y_cursor - header_h_px) / fig_h_px,
            hm_w_px / fig_w_px,
            header_h_px / fig_h_px
        ])
        header_ax.set_xlim(-0.5, len(ENTITIES) - 0.5)
        header_ax.set_ylim(0, 1)
        header_ax.axis("off")
        for xi, name in enumerate(ENTITIES):
            header_ax.text(xi, 0.5, name, ha="center", va="center", fontsize=XTICK_FSIZE)
        y_cursor -= header_h_px

        # TOTAL block (INSIDE H1 box)
        y_cursor -= TOTAL_BLOCK_TOP_PAD_PX
        total_label = f"{p['title']} total - {p['total_pct']}"
        ax_lbl_t = fig.add_axes([
            x0_col / fig_w_px,
            (y_cursor - label_h_px) / fig_h_px,
            col_w_px / fig_w_px,
            label_h_px / fig_h_px
        ])
        ax_lbl_t.axis("off")
        ax_lbl_t.text(0.5, 0.5, total_label, ha="center", va="center", fontsize=LABEL_FSIZE)
        y_cursor -= label_h_px + LABEL_TO_HM_GAP_PX

        ax_hm_t = fig.add_axes([
            x0_hm / fig_w_px,
            (y_cursor - hm_h_px) / fig_h_px,
            hm_w_px / fig_w_px,
            hm_h_px / fig_h_px
        ])
        # TOTAL row as rounded cells
        draw_heatmap_row(ax_hm_t, p["macro_row"], hm_w_px, hm_h_px, cmap, norm)

        # After TOTAL heatmap: default item gap + stretch (1 slot)
        y_cursor -= hm_h_px + ITEM_GAP_PX + stretch

        # ── Meso groups (INSIDE H1 box), each in a wider rounded meso box ────
        for gi, g in enumerate(p["meso_groups"]):
            n_rows = len(g["rows"])
            if n_rows == 0:
                continue

            group_box_h = group_geom[gi]
            # Meso box is wider than column
            meso_box_x = x0_col - MESO_BOX_OUTSET_PX
            meso_box_w = col_w_px + 2 * MESO_BOX_OUTSET_PX

            # Axes that match meso box in pixels → 1 data unit = 1 px
            box_ax = fig.add_axes([
                meso_box_x / fig_w_px,
                (y_cursor - group_box_h) / fig_h_px,
                meso_box_w / fig_w_px,
                group_box_h / fig_h_px
            ])
            meso_box_w_px = meso_box_w
            group_box_h_px = group_box_h
            box_ax.set_xlim(0, meso_box_w_px)
            box_ax.set_ylim(0, group_box_h_px)
            box_ax.axis("off")

            # Rounded meso box
            meso_patch = FancyBboxPatch(
                (0, 0), meso_box_w_px, group_box_h_px,
                boxstyle=f"round,pad=0.0,rounding_size={MESO_BOX_RADIUS_PX}",
                transform=box_ax.transData,
                facecolor=MESO_FACE_COLOR,
                edgecolor=MESO_EDGE_COLOR,
                linewidth=1.0,
                alpha=MESO_FACE_ALPHA,
                zorder=0,
                antialiased=True
            )
            box_ax.add_patch(meso_patch)

            # Centered meso title at top (inside box), using axes coords
            box_ax.text(
                0.5, 1 - (MESO_PAD_PX / group_box_h_px),
                g["meso"], ha="center", va="top",
                fontsize=MESO_FSIZE, color="black", alpha=0.95,
                transform=box_ax.transAxes
            )

            # Rows INSIDE meso box: start below title + inner padding + extra title gap
            y_rows = y_cursor - MESO_PAD_PX - meso_title_h_px - MESO_TITLE_BOTTOM_GAP_PX

            # Draw rows; apply stretch after each row EXCEPT the last in the meso
            for local_i, r in enumerate(g["rows"]):
                last_in_group = (local_i == len(g["rows"]) - 1)
                label_str = f"{p['micro_labels'][r]} - {p['pct_labels'][r]}"

                ax_lbl = fig.add_axes([
                    x0_col / fig_w_px,
                    (y_rows - label_h_px) / fig_h_px,
                    col_w_px / fig_w_px,
                    label_h_px / fig_h_px
                ])
                ax_lbl.axis("off")
                ax_lbl.text(0.5, 0.5, label_str, ha="center", va="center", fontsize=LABEL_FSIZE)
                y_rows -= label_h_px + LABEL_TO_HM_GAP_PX

                ax_hm = fig.add_axes([
                    x0_hm / fig_w_px,
                    (y_rows - hm_h_px) / fig_h_px,
                    hm_w_px / fig_w_px,
                    hm_h_px / fig_h_px
                ])
                row_vals = p["matrices"][r, :]
                draw_heatmap_row(ax_hm, row_vals, hm_w_px, hm_h_px, cmap, norm)

                # Gap after row: default + stretch unless it's the last row in this meso
                y_rows -= hm_h_px + ITEM_GAP_PX + (0.0 if last_in_group else stretch)

            # advance y_cursor by full meso box height
            y_cursor -= group_box_h
            # inter-meso gap (if not last meso): default + stretch
            if gi < len(p["meso_groups"]) - 1:
                y_cursor -= MESO_BOX_GAP_PX + stretch

        # bottom inner padding of H1 column box
        y_cursor -= COL_BOX_PAD_PX

    # ── Global horizontal colorbar (height = heatmap height) ──────────────────
    panels_left_px = left_px
    panels_right_px = left_px + ncols * col_w_px + (ncols - 1) * COL_GAP_PX
    panels_span_px = panels_right_px - panels_left_px

    cbar_width_frac = 0.50
    cbar_left_px = panels_left_px + (1.0 - cbar_width_frac) * 0.5 * panels_span_px
    cbar_width_px = cbar_width_frac * panels_span_px

    cbar_height_px = hm_h_px
    cbar_bottom_px = max(bot_y_px - cbar_height_px - 0.03 * fig_h_px, 0.01 * fig_h_px)

    cax = fig.add_axes([
        cbar_left_px / fig_w_px,
        cbar_bottom_px / fig_h_px,
        cbar_width_px / fig_w_px,
        cbar_height_px / fig_h_px
    ])

    def cbar_formatter(v, pos):
        if abs(v - 0.5) < 1e-6:
            return "50%+"
        return f"{v * 100:.0f}%"

    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=cmap, norm=norm,
        orientation="horizontal", ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    cb.ax.tick_params(labelsize=CBAR_FSIZE)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(cbar_formatter))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_SVG), exist_ok=True)
    fig.savefig(OUT_SVG, format="svg", bbox_inches="tight")
    print(f"✅ Saved: {OUT_SVG}")
    print("Items per macro:", [p['matrices'].shape[0] for p in panels])
    print(
        "Layout: column gap = {}px, H1 box outset = {}px, H2 box outset = {}px; fonts scaled by {}×."
        .format(COL_GAP_PX, COL_BOX_OUTSET_PX, MESO_BOX_OUTSET_PX, SCALE)
    )


if __name__ == "__main__":
    main()