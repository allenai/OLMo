"""

Plotting the FLOPS by performance figure

Don't forget to run `pip install -e '.[figures]'` to install the necessary dependencies.

@kylel


Sample CSV file looks like:
```
Model,FLOPs,Average,ARC Challenge,HSwag,WinoG,MMLU,DROP,NQ,AGIEval,GSM8k,MMLU Pro,TriviaQA
Amber-7B,5.091E+22,35.2,44.9,74.5,65.5,24.7,26.1,18.7,21.8,4.8,11.7,59.3
DCLM-7B,1.033E+23,56.9,79.8,82.3,77.3,64.4,39.3,28.8,47.5,46.1,31.3,72.1
Gemma-2-9B,4.436E+23,67.8,89.5,87.3,78.8,70.6,63,38,57.3,70.1,42,81.8
Llama-2-13B,1.562E+23,54.1,67.3,83.9,74.9,55.7,45.6,38.4,41.5,28.1,23.9,81.3
Llama-3.1-8B,7.227E+23,61.8,79.5,81.6,76.6,66.9,56.4,33.9,51.3,56.5,34.7,80.3
MAP-Neo-7B,2.106E+23,49.6,78.4,72.8,69.2,58,39.4,28.9,45.8,12.5,25.9,65.1
Mistral-7B-v0.3,,58.8,78.3,83.1,77.7,63.5,51.8,37.2,47.3,40.1,30,79.3
Mistral-Nemo-Bs-12B,,66.9,85.2,85.6,81.5,69.5,69.2,39.7,54.7,62.1,36.7,84.6
OLMo-0424-7B,8.679E+22,50.7,66.9,80.1,73.6,54.3,50,29.6,43.9,27.7,22.1,58.8
OLMo-2-1124-13B,4.609E+23,68.3,83.5,86.4,81.5,67.5,70.7,46.7,54.2,75.1,35.1,81.9
OLMo-2-1124-7B,1.771E+23,62.9,79.8,83.8,77.2,63.7,60.8,36.9,50.4,67.5,31,78
OLMo-7B,1.018E+23,38.3,46.4,78.1,68.5,28.3,27.3,24.8,23.7,9.2,12.1,64.1
Qwen-2.5-14B,1.595E+24,72.2,94.0,94,80,79.3,51.5,37.3,71,83.4,52.8,79.1
Qwen-2.5-7B,8.225E+23,67.4,89.5,89.7,74.2,74.4,55.8,29.9,63.7,81.5,45.8,69.4
StableLM-2-12B,2.929E+23,62.2,81.9,84.5,77.7,62.4,55.5,37.6,50.9,62,29.3,79.9
Zamba-2-7B,,65.2,92.2,89.4,79.6,68.5,51.7,36.5,55.5,67.2,32.8,78.8
```


"""

# First clear any existing font cache
import shutil

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Find path to Manrope font on your computer
# fonts = [f for f in fm.findSystemFonts()]
# manrope_fonts = [f for f in fonts if "manrope" in f.lower()]
# for font in manrope_fonts:
#     print(font)


# Load the font file
font_path = "/Users/kylel/Library/Fonts/Manrope-VariableFont_wght.ttf"
font_prop = FontProperties(fname=font_path)


# Try setting weight after creation
font_prop.set_weight(500)


# Set it globally using the font property
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_prop.get_name()]

import numpy as np
import pandas as pd

RESULTS_DATA_PATH = "/Users/kylel/ai2/olmo/olmo2.csv"
OUTPUT_PATHS = ["/Users/kylel/ai2/olmo/olmo2.pdf", "/Users/kylel/ai2/olmo/olmo2.png"]
df = pd.read_csv(RESULTS_DATA_PATH)

# don't count Model, Flops, and Average columns
num_datasets = len(df.columns) - 3

MODEL_COLUMN_NAME = "Model"
CATEGORY_COLUMN_NAME = "category"
FLOPS_COLUMN_NAME = "FLOPs"
METRIC_COLUMN_NAME = "Average"
COLOR_COLUMN_NAME = "color"
OFFSET_COLUMN_NAME = "label_offset"
MARKER_COLUMN_NAME = "marker"

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

df[COLOR_COLUMN_NAME] = df[CATEGORY_COLUMN_NAME].map(category_to_color)

model_name_to_label_offset = {
    "Amber-7B": [10, -2],  # Move left and down to use empty space
    "DCLM-7B": [-20, 10],  # Move right and up into empty area
    "Mistral-7B-v0.3": [-20, 8],  # Move left and up
    "Mistral-Nemo-Bs-12B": [20, -8],  # Move right and down
    "Gemma-2-9B": [-35, -15],  # Move left and down
    "Llama-2-13B": [0, 10],  # Move right and slightly up
    "Llama-3.1-8B": [0, -15],  # Move right and down
    "MAP-Neo-7B": [-20, -15],  # Move left and down into empty space
    "Zamba-2-7B": [-25, 10],  # Move left and up
    "OLMo-0424-7B": [-35, -15],  # Move right and slightly down
    "OLMo-2-1124-13B": [-20, 10],  # Move left and up
    "OLMo-2-1124-7B": [-35, 10],  # Move right
    "OLMo-7B": [-35, 10],  # Move left and up into empty space
    "Qwen-2.5-14B": [-40, -15],  # Move right and up
    "Qwen-2.5-7B": [-20, -15],  # Move left and down
    "StableLM-2-12B": [0, -15],  # Move right and up
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
# rename models
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
# plt.xscale("log")
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
        c=data[COLOR_COLUMN_NAME],  # Use the colors column
        marker=category_to_marker[category],  # Add the marker parameter
        alpha=category_to_alpha[category],
        s=category_to_marker_size[category],
    )
# Add labels for each point
FONTSIZE = 9
for idx, row in df[df[FLOPS_COLUMN_NAME].notna()].iterrows():
    plt.annotate(
        row[MODEL_COLUMN_NAME],
        (row[FLOPS_COLUMN_NAME], row[METRIC_COLUMN_NAME]),
        xytext=(row[OFFSET_COLUMN_NAME]),
        textcoords="offset points",
        fontsize=FONTSIZE,
        alpha=1.0,
    )

# x axis tick marks
tick_locations = [4e22, 6e22, 8e22, 1e23, 2e23, 4e23, 6e23, 8e23, 1e24, 2e24]


# Function to format numbers in scientific notation (10^x) more intuitively
def format_scientific(x):
    exponent = int(np.log10(x))  # Get the exponent
    mantissa = x / (10**exponent)  # Get the mantissa

    # Format as "1×10²²", "2×10²²", etc.
    return f"{int(mantissa)}×10{str(exponent).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


tick_labels = [format_scientific(x) for x in tick_locations]
plt.xticks(tick_locations, tick_labels, rotation=45, ha="right")


# Customize the plot
plt.xlabel("Approximate FLOPs", fontsize=12)
plt.ylabel(f"Avg Performance ({num_datasets} Benchmarks)", fontsize=12)

# Add grid
plt.grid(True, which="both", ls="-", alpha=0.2)

# Customize legend
# plt.legend(title="", title_fontsize=10, fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")

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
    fontsize=8,
    handletextpad=0.05,  # Reduce space between marker and label
    columnspacing=0.5,  # Adjust space between columns
    frameon=False,  # Remove the legend border
)

# Adjust the layout to prevent legend cutoff
plt.tight_layout()
plt.subplots_adjust(top=0.8)  # Make room for the legend


# Remove spines
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Make Yellow portion
xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()

# Assuming df has columns 'x' and 'y' and is sorted by x
# Convert frontier points to polygon vertices
frontier_models = ["Amber-7B", "OLMo-0424-7B", "DCLM-7B", "OLMo-2-7B", "OLMo-2-13B", "Qwen-2.5-14B"]
frontier_df = df[df[MODEL_COLUMN_NAME].isin(frontier_models)]
frontier_df = frontier_df.set_index(MODEL_COLUMN_NAME)
frontier_df = frontier_df.reindex(frontier_models)
frontier_df = frontier_df.reset_index()

# Create simple vertices array:
X = np.array([[xmin, ymin]])  # Start bottom-left
for _, row in frontier_df.iterrows():
    X = np.append(X, [[row[FLOPS_COLUMN_NAME], row[METRIC_COLUMN_NAME]]], axis=0)
X = np.append(X, [[xmax, ymax]], axis=0)  # Top-right corner
X = np.append(X, [[xmin, ymax]], axis=0)  # Back to left

# Create and add polygon
polygon = plt.Polygon(X, facecolor="yellow", alpha=0.2, zorder=-1, edgecolor="orange", linestyle="--", linewidth=2)
plt.gca().add_patch(polygon)

# Save the figure
for output_path in OUTPUT_PATHS:
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
