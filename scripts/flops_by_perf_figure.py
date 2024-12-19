import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
from matplotlib import font_manager
from cached_path import cached_path


ap = argparse.ArgumentParser()
ap.add_argument("results_data_path", type=str, help="Path to the results data CSV file.")
ap.add_argument("output_dir", type=str, help="Path to the output directory")
ap.add_argument("--manrope-medium-font-path", type=str, help="Path to the Manrope Medium font file", default="https://dolma-artifacts.org/Manrope-Medium.ttf")
args = ap.parse_args()

# Add Manrope font
font_manager.fontManager.addfont(cached_path(args.manrope_medium_font_path))
plt.rcParams['font.family'] = 'Manrope'
plt.rcParams['font.weight'] = 'medium'


OUTPUT_PATHS = [f"{args.output_dir}/olmo2.pdf", f"{args.output_dir}/olmo2.png"]
df = pd.read_csv(args.results_data_path)

# don't count Model, Flops, and Average columns
num_datasets = len(df.columns) - 3

MODEL_COLUMN_NAME = "Model"
CATEGORY_COLUMN_NAME = "category"
FLOPS_COLUMN_NAME = "FLOPs"
METRIC_COLUMN_NAME = "Average"
COLOR_COLUMN_NAME = "color"
OFFSET_COLUMN_NAME = "label_offset"
MARKER_COLUMN_NAME = "marker"

AI2_YELLOW = "#fff500"
AI2_ORANGE = "#f65834"
AI2_DARK_TEAL = "#0a3235"
AI2_OFF_WHITE = "#faf2e9"
AI2_TEAL = "#105257"
AI2_PINK = "#f0529c"
AI2_PURPLE = "#b11be8"
AI2_GREEN = "#0fcb8c"

# remove Zamba model (SSM, not a language model)
df = df[df[MODEL_COLUMN_NAME] != "Zamba-2-7B"]

model_name_to_open_status = {
    "Amber-7B": "Other fully open",
    "DCLM-7B": "Other fully open",
    "Mistral-7B-v0.3": "Open weights",
    "Mistral-Nemo-Bs-12B": "Open weights",
    "Gemma-2-9B": "Open weights",
    "Llama-2-13B": "Open weights",
    "Llama-3.1-8B": "Open weights",
    "MAP-Neo-7B": "Other fully open",
    "Zamba-2-7B": "Partially open",
    "OLMo-0424-7B": "Previous OLMo",
    "OLMo-2-1124-13B": "Latest OLMo",
    "OLMo-2-1124-7B": "Latest OLMo",
    "OLMo-2-13B": "Latest OLMo",
    "OLMo-2-7B": "Latest OLMo",
    "OLMo-7B": "Previous OLMo",
    "Qwen-2.5-14B": "Open weights",
    "Qwen-2.5-7B": "Open weights",
    "StableLM-2-12B": "Partially open",
}

# Add a column for model category based on the groupings
df[CATEGORY_COLUMN_NAME] = df[MODEL_COLUMN_NAME].map(model_name_to_open_status)

# Add a column for color based on the category
categories = df["category"].unique()
category_to_color = {
    "Open weights": "#093235",  # dark blue
    "Partially open": "#255457",  # dark green
    "Other fully open": "#6FE0BA",  # light green
    "Previous OLMo": "#F697C4",  # light pink
    "Latest OLMo": "#F0529C",  # dark pink
}
category_to_text_color = {
    "Open weights": AI2_DARK_TEAL,
    "Partially open": AI2_DARK_TEAL,
    "Other fully open": AI2_DARK_TEAL,
    "Previous OLMo": AI2_DARK_TEAL,
    "Latest OLMo": "#a51c5c",   # darker pink
}


df[COLOR_COLUMN_NAME] = df[CATEGORY_COLUMN_NAME].map(category_to_color)

model_name_to_label_offset = {
    "Amber-7B": [10, -2],
    "DCLM-7B": [-18, 8],
    "Mistral-7B-v0.3": [-20, 8],
    "Mistral-Nemo-Bs-12B": [20, -8],
    "Gemma-2-9B": [-35, -15],
    "Llama-2-13B": [-5, 7],
    "Llama-3.1-8B": [-20, -13],
    "MAP-Neo-7B": [-20, -15],
    "Zamba-2-7B": [-25, 10],
    "OLMo-0424-7B": [-35, -15],
    "OLMo-2-1124-13B": [-20, 10],
    "OLMo-2-1124-7B": [-35, 10],
    "OLMo-7B": [-15, 10],
    "Qwen-2.5-14B": [-40, -15],
    "Qwen-2.5-7B": [-20, -15],
    "StableLM-2-12B": [-20, -15],
}

df[OFFSET_COLUMN_NAME] = df[MODEL_COLUMN_NAME].map(model_name_to_label_offset)

# markers
category_to_marker = {
    "Open weights": "o",
    "Partially open": "D",
    "Other fully open": "s",
    "Previous OLMo": "P",
    "Latest OLMo": "*",
}

# Clean up labels
model_name_to_new_name = {
    "OLMo-2-1124-13B": "OLMo-2-13B",
    "OLMo-2-1124-7B": "OLMo-2-7B",
}
df[MODEL_COLUMN_NAME] = df[MODEL_COLUMN_NAME].replace(model_name_to_new_name)

# marker size
category_to_marker_size = {
    "Open weights": 40,
    "Partially open": 40,
    "Other fully open": 70,
    "Previous OLMo": 100,
    "Latest OLMo": 150,
}

# alpha
category_to_alpha = {
    "Open weights": 1.0,
    "Partially open": 0.7,
    "Other fully open": 1.0,
    "Previous OLMo": 1.0,
    "Latest OLMo": 1.0,
}

# Scale
plt.xscale("function", functions=(np.sqrt, np.square))

# Plotting order
desired_order = ["Latest OLMo", "Previous OLMo", "Other fully open", "Partially open", "Open weights"]
for category in categories:
    mask = (df[CATEGORY_COLUMN_NAME] == category) & (df[FLOPS_COLUMN_NAME].notna())
    data = df[mask]
    plt.scatter(
        data[FLOPS_COLUMN_NAME],
        data[METRIC_COLUMN_NAME],
        label=category,
        c=data[COLOR_COLUMN_NAME],
        marker=category_to_marker[category],
        alpha=category_to_alpha[category],
        s=category_to_marker_size[category],
    )

# Add labels for each point with Manrope Medium
FONTSIZE = 9
for idx, row in df[df[FLOPS_COLUMN_NAME].notna()].iterrows():
    plt.annotate(
        row[MODEL_COLUMN_NAME],
        (row[FLOPS_COLUMN_NAME], row[METRIC_COLUMN_NAME]),
        xytext=(row[OFFSET_COLUMN_NAME]),
        textcoords="offset points",
        fontsize=FONTSIZE,
        alpha=1.0,
        font='Manrope',
        weight='medium',
        color=category_to_text_color[model_name_to_open_status[row[MODEL_COLUMN_NAME]]],
    )

# x axis tick marks
tick_locations = [4e22, 6e22, 8e22, 1e23, 2e23, 4e23, 6e23, 8e23, 1e24, 2e24]

def format_scientific(x):
    exponent = int(np.log10(x))
    mantissa = x / (10**exponent)
    return f"{int(mantissa)}×10{str(exponent).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"

tick_labels = [format_scientific(x) for x in tick_locations]
plt.xticks(tick_locations, tick_labels, rotation=45, ha="right", fontsize=8)

# y axis tick marks
plt.yticks(fontsize=8)

# Customize the plot with Manrope Medium
plt.xlabel("Approximate FLOPs", fontsize=10, font='Manrope', weight='medium')
plt.ylabel(f"Avg Performance ({num_datasets} Benchmarks)", fontsize=10, font='Manrope', weight='medium')


# Add grid with custom colors
plt.grid(True, which="major", ls=":", color="#105257", alpha=0.2)
plt.grid(True, which="minor", ls="-", color="#9fbabc", alpha=0.2)

# Also set the tick colors
plt.tick_params(which='major', colors='#105257')
plt.tick_params(which='minor', colors='#9fbabc')

# If you want to change the actual axis line colors as well
plt.gca().spines['left'].set_color('#105257')
plt.gca().spines['bottom'].set_color('#105257')

# Add the legend below the plot
handles, labels = plt.gca().get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))
ordered_handles = [label_to_handle[label] for label in desired_order]
plt.legend(
    ordered_handles,
    desired_order,
    bbox_to_anchor=(0, 0.97, 1.0, 0.2),
    loc="center",
    ncol=len(categories),
    mode="expand",
    borderaxespad=0.0,
    fontsize=6,
    handletextpad=0.05,
    columnspacing=0.5,
    frameon=False,
    prop={'family': 'Manrope', 'weight': 'medium', 'size': 8}
)

# Adjust the layout
plt.tight_layout()
plt.subplots_adjust(top=0.8)

# Remove spines
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Make Yellow portion
xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()

# Convert frontier points to polygon vertices
frontier_models = ["Amber-7B", "OLMo-0424-7B", "DCLM-7B", "OLMo-2-7B", "OLMo-2-13B", "Qwen-2.5-14B"]
frontier_df = df[df[MODEL_COLUMN_NAME].isin(frontier_models)]
frontier_df = frontier_df.set_index(MODEL_COLUMN_NAME)
frontier_df = frontier_df.reindex(frontier_models)
frontier_df = frontier_df.reset_index()

# in order for the line not to appear at the top of the polygon, we need to offset it
polygon_line_width = 1
polygon_offset = (ymax - ymin) * (polygon_line_width / 100)

# Create simple vertices array
X = np.array([[xmin, ymin]])  # Start bottom-left
for _, row in frontier_df.iterrows():
    X = np.append(X, [[row[FLOPS_COLUMN_NAME], row[METRIC_COLUMN_NAME]]], axis=0)
X = np.append(X, [[xmax, ymax + polygon_offset]], axis=0)  # Top-right corner
X = np.append(X, [[xmin, ymax + polygon_offset]], axis=0)  # Back to left

# Create and add polygon
polygon = plt.Polygon(X, facecolor=AI2_YELLOW, alpha=0.2, zorder=-1, edgecolor=AI2_ORANGE, linestyle="--", linewidth=1.5)
plt.gca().add_patch(polygon)

# Save the figure
for output_path in OUTPUT_PATHS:
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
