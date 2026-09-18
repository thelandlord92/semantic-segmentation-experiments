from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# CONFIG
# ============================================================

# ------------------------------------------------------------
# FILE
# ------------------------------------------------------------

parent_dir = Path(__file__).parent.parent.parent

excel_path = (
    parent_dir
    / "data"
    / "excel"
    / "hxe_sam_evaluation_sheet.xlsx"
)

sheet_name = "Class-wise Metrics"


# ------------------------------------------------------------
# CLASS FILTER
# ------------------------------------------------------------

# Options:
# "timber beams"
# "timber columns"

class_filter = "timber beams"


# ------------------------------------------------------------
# AXIS LABELS
# ------------------------------------------------------------

x_axis_label = "Ground Truth Count"
y_axis_label = "SAM3 True Positive Count"


# ------------------------------------------------------------
# PLOT TITLE
# ------------------------------------------------------------

plot_title = f"SAM3 Performance: {class_filter}"


# ------------------------------------------------------------
# DATA POINT APPEARANCE
# ------------------------------------------------------------

# Common marker options:
# "o" = circle
# "s" = square
# "^" = triangle
# "D" = diamond
# "x" = cross
# "+" = plus

marker_style = "o"
marker_size = 40
marker_alpha = 0.8

marker_edge_color = "black"
marker_edge_width = 0.5


# ------------------------------------------------------------
# RECALL COLOUR SETTINGS
# ------------------------------------------------------------

# Examples:
# "viridis"
# "plasma"
# "inferno"
# "magma"
# "cividis"
# "coolwarm"

recall_colormap = "viridis"

recall_min = 0.0
recall_max = 1.0

recall_legend_label = "Recall"


# ------------------------------------------------------------
# PERFECT-PERFORMANCE REFERENCE LINE
# ------------------------------------------------------------

show_reference_line = True

reference_line_style = "--"
reference_line_width = 1.5
reference_line_color = "black"
reference_line_label = "Perfect detection"


# ------------------------------------------------------------
# GRAPH APPEARANCE
# ------------------------------------------------------------

figure_width = 8
figure_height = 8

show_grid = True
grid_alpha = 0.25

axis_font_size = 12
title_font_size = 14

show_legend = True


# ------------------------------------------------------------
# AXIS SETTINGS
# ------------------------------------------------------------

# Maintain the same scale on both axes.
equal_axis_scale = True

# Padding beyond the maximum plotted value.
axis_padding = 1


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

save_plot = True

output_dir = (
    parent_dir
    / "results"
    / "sam3_evaluation_plots"
)

output_file_name = (
    class_filter.replace(" ", "_")
    + "_gt_vs_true_positives.png"
)

plot_dpi = 300


# ============================================================
# LOAD EXCEL DATA
# ============================================================

df = pd.read_excel(
    excel_path,
    sheet_name=sheet_name,
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_name",
    "class",
    "gt_count",
    "true_positives",
    "recall",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# FILTER BY CLASS
# ============================================================

class_df = df[
    df["class"] == class_filter
].copy()

if class_df.empty:
    raise ValueError(
        f"No rows found for class: {class_filter}"
    )


# ============================================================
# REMOVE ROWS WITHOUT VALUES
# ============================================================

plot_df = class_df.dropna(
    subset=[
        "gt_count",
        "true_positives",
        "recall",
    ]
).copy()

if plot_df.empty:
    raise ValueError(
        f"No rows for '{class_filter}' contain valid "
        "ground-truth, true-positive, and recall values."
    )


# ============================================================
# CONVERT TO NUMERIC VALUES
# ============================================================

numeric_columns = [
    "gt_count",
    "true_positives",
    "recall",
]

for column in numeric_columns:
    plot_df[column] = pd.to_numeric(
        plot_df[column],
        errors="coerce",
    )

plot_df = plot_df.dropna(
    subset=numeric_columns
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        figure_width,
        figure_height,
    ),
    layout="constrained",
)


# ============================================================
# SCATTER PLOT
# ============================================================

scatter = ax.scatter(
    plot_df["gt_count"],
    plot_df["true_positives"],
    c=plot_df["recall"],
    cmap=recall_colormap,
    vmin=recall_min,
    vmax=recall_max,
    marker=marker_style,
    s=marker_size,
    alpha=marker_alpha,
    edgecolors=marker_edge_color,
    linewidths=marker_edge_width,
)


# ============================================================
# RECALL COLOUR BAR
# ============================================================

colorbar_ax = inset_axes(
    ax,
    width="3%",
    height="45%",
    loc="center right",
    borderpad=-4,
)

colorbar = fig.colorbar(
    scatter,
    cax=colorbar_ax,
)

colorbar.set_label(
    recall_legend_label,
    fontsize=axis_font_size,
)


# ============================================================
# REFERENCE LINE
# ============================================================

max_value = max(
    plot_df["gt_count"].max(),
    plot_df["true_positives"].max(),
)

axis_max = max_value + axis_padding

if show_reference_line:
    ax.plot(
        [0, axis_max],
        [0, axis_max],
        linestyle=reference_line_style,
        linewidth=reference_line_width,
        color=reference_line_color,
        label=reference_line_label,
    )


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    x_axis_label,
    fontsize=axis_font_size,
)

ax.set_ylabel(
    y_axis_label,
    fontsize=axis_font_size,
)

ax.set_title(
    plot_title,
    fontsize=title_font_size,
)

ax.set_xlim(
    0,
    axis_max,
)

ax.set_ylim(
    0,
    axis_max,
)

if equal_axis_scale:
    ax.set_aspect(
        "equal",
        adjustable="box",
    )


# ============================================================
# GRID
# ============================================================

if show_grid:
    ax.grid(
        True,
        alpha=grid_alpha,
    )


# ============================================================
# LEGEND
# ============================================================

if show_legend and show_reference_line:
    ax.legend()


# ============================================================
# SAVE
# ============================================================

if save_plot:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / output_file_name
    )

    plt.savefig(
        output_path,
        dpi=plot_dpi,
        bbox_inches="tight",
    )

    print(
        f"Saved plot: {output_path}"
    )


# ============================================================
# SHOW
# ============================================================

plt.show()