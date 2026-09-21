import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# USER PARAMETERS
# ============================================================

parent_dir = Path(__file__).parent.parent.parent

# Input / output
VAL_STATS_PATH = (
    parent_dir /
    "data" /
    "training_and_evaluation_data" /
    "hxe" /
    "v1" /
    "val_stats.json"
)

OUTPUT_PATH = (
    parent_dir /
    "results" /
    "sam_evaluation_plots" /
    "average_precision_plot.png"
)

# Checkpoint selection
SELECTED_EPOCH = 13
SHOW_SELECTED_EPOCH = True
SHOW_BEST_AP_MARKER = True

# Figure
FIGURE_WIDTH = 8.0
FIGURE_HEIGHT = 5.0
DPI = 300

# Labels
TITLE = "Validation Segmentation Performance"
X_LABEL = "Epoch"
Y_LABEL = "Average Precision"

# Font sizes
TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 9

# Lines
LINE_WIDTH = 2.0
MARKER_SIZE = 5
AP_MARKER = "o"
AP50_MARKER = "s"
AP75_MARKER = "^"

# Selected epoch line
SELECTED_LINE_STYLE = "--"
SELECTED_LINE_WIDTH = 1.5
SELECTED_LINE_ALPHA = 0.8

# Axis
X_TICK_INTERVAL = 1
Y_MIN = 0.0
Y_MAX = 0.8

# Grid
SHOW_GRID = True
GRID_ALPHA = 0.25
GRID_LINE_STYLE = "--"

# Legend
LEGEND_LOCATION = "lower right"

# Output
SAVE_FIGURE = True
SHOW_FIGURE = True


# ============================================================
# METRIC KEYS
# ============================================================

AP_KEY = (
    "Meters_train/val_timber/segmentation/"
    "coco_eval_segm_AP"
)
AP50_KEY = (
    "Meters_train/val_timber/segmentation/"
    "coco_eval_segm_AP_50"
)
AP75_KEY = (
    "Meters_train/val_timber/segmentation/"
    "coco_eval_segm_AP_75"
)
EPOCH_KEY = "Trainer/epoch"


# ============================================================
# FUNCTIONS
# ============================================================

def load_json_lines(file_path):
    """Load a JSON-lines statistics file."""
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def extract_metrics(records):
    """Extract epoch and validation AP metrics."""
    epochs = []
    ap_values = []
    ap50_values = []
    ap75_values = []

    for record in records:
        if AP_KEY not in record:
            continue

        epochs.append(record[EPOCH_KEY])
        ap_values.append(record[AP_KEY])
        ap50_values.append(record[AP50_KEY])
        ap75_values.append(record[AP75_KEY])

    return epochs, ap_values, ap50_values, ap75_values


# ============================================================
# LOAD DATA
# ============================================================

records = load_json_lines(VAL_STATS_PATH)

epochs, ap_values, ap50_values, ap75_values = extract_metrics(
    records
)

best_index = max(
    range(len(ap_values)),
    key=ap_values.__getitem__,
)

best_epoch = epochs[best_index]
best_ap = ap_values[best_index]

print(f"Best validation AP: {best_ap:.4f}")
print(f"Best validation epoch: {best_epoch}")


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    layout="constrained",
)

ax.plot(
    epochs,
    ap_values,
    marker=AP_MARKER,
    markersize=MARKER_SIZE,
    linewidth=LINE_WIDTH,
    label="AP",
)

ax.plot(
    epochs,
    ap50_values,
    marker=AP50_MARKER,
    markersize=MARKER_SIZE,
    linewidth=LINE_WIDTH,
    label="AP50",
)

ax.plot(
    epochs,
    ap75_values,
    marker=AP75_MARKER,
    markersize=MARKER_SIZE,
    linewidth=LINE_WIDTH,
    label="AP75",
)

if SHOW_SELECTED_EPOCH:
    ax.axvline(
        x=SELECTED_EPOCH,
        linestyle=SELECTED_LINE_STYLE,
        linewidth=SELECTED_LINE_WIDTH,
        alpha=SELECTED_LINE_ALPHA,
        label=f"Selected epoch ({SELECTED_EPOCH})",
    )

if SHOW_BEST_AP_MARKER:
    ax.scatter(
        best_epoch,
        best_ap,
        s=MARKER_SIZE**2 * 3,
        zorder=5,
    )

    ax.annotate(
        f"Best AP = {best_ap:.3f}",
        xy=(best_epoch, best_ap),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=ANNOTATION_FONT_SIZE,
    )

ax.set_title(
    TITLE,
    fontsize=TITLE_FONT_SIZE,
)

ax.set_xlabel(
    X_LABEL,
    fontsize=AXIS_LABEL_FONT_SIZE,
)

ax.set_ylabel(
    Y_LABEL,
    fontsize=AXIS_LABEL_FONT_SIZE,
)

ax.set_ylim(Y_MIN, Y_MAX)

ax.set_xticks(
    range(
        min(epochs),
        max(epochs) + 1,
        X_TICK_INTERVAL,
    )
)

ax.tick_params(
    axis="both",
    labelsize=TICK_FONT_SIZE,
)

if SHOW_GRID:
    ax.grid(
        alpha=GRID_ALPHA,
        linestyle=GRID_LINE_STYLE,
    )

ax.legend(
    loc=LEGEND_LOCATION,
    fontsize=LEGEND_FONT_SIZE,
)

if SAVE_FIGURE:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=DPI,
        bbox_inches="tight",
    )

if SHOW_FIGURE:
    plt.show()

plt.close(fig)