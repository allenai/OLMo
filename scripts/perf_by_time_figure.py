"""


Data looks like:
    Model,Average,Class,Date
    GPT-3.5 Turbo 0125,59.6,Closed API model,Jan-24
    GPT 4o mini 0718,65.7,Closed API model,Jul-24
    Gemma 2 27B,61.3,"Open Weights model, 24-32B",Jun-24
    Qwen2.5-32B,66.5,"Open Weights model, 24-32B",Sep-24
    Mistral Small 24B,67.6,"Open Weights model, 24-32B",Jan-25
    Qwen QwQ 32B,,"Open Weights model, 24-32B",Mar-25
    Gemma 3 27B,,"Open Weights model, 24-32B",Mar-25
    Qwen 2.5 72B,68.8,"Open Weights model, 70B+",Sep-24
    Llama 3.1 70B,70,"Open Weights model, 70B+",Jul-24
    Llama 3.3 70B,73,"Open Weights model, 70B+",Dec-24
    OLMo 2 7B,55.7,Fully Open Models,Nov-24
    OLMo 2 13B,61.4,Fully Open Models,Nov-24
    OLMo 2 32B,68.8,Fully Open Models,Mar-25

Commands:
    python scripts/perf_by_time_figure.py -i olmo2-32b-timeline.csv -o output/timeline.png

@lucas, @kylel

"""

import os
from datetime import datetime
from io import BytesIO

import click
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D


# Function to parse the date string into a datetime object
def parse_date(date_str):
    if isinstance(date_str, str) and "-" in date_str:
        # Format like "Jan-24"
        month_map = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }
        month, year = date_str.split("-")
        return datetime(2000 + int(year), month_map[month], 1)
    else:
        # Try to parse directly
        return pd.to_datetime(date_str)


@click.command()
@click.option("--input-path", "-i", required=True, help="Path to input CSV file")
@click.option("--output-path", "-o", help="Path to save the output image", default=None)
@click.option(
    "--manrope-font-path",
    help="Path to the Manrope Medium font file",
    default="https://dolma-artifacts.org/Manrope-Medium.ttf",
)
def main(input_path, output_path, manrope_font_path):
    """Generate a scatter plot from model performance CSV data."""

    # Setup custom Manrope font
    try:
        # Check if it's a URL or a local file
        if manrope_font_path.startswith(("http://", "https://")):
            # Download the font if it's a URL
            response = requests.get(manrope_font_path)
            response.raise_for_status()  # Raise exception for HTTP errors
            font_file = BytesIO(response.content)
            font_path = font_manager.fontManager.addfont(font_file)
            click.echo(f"Downloaded and added Manrope font from URL")
        else:
            # Use local file path
            font_path = font_manager.fontManager.addfont(manrope_font_path)
            click.echo(f"Added Manrope font from local path")

        # Set as default font
        plt.rcParams["font.family"] = "Manrope"
        plt.rcParams["font.weight"] = "medium"
    except Exception as e:
        click.echo(f"Warning: Could not load Manrope font: {e}")
        click.echo("Continuing with default font...")

    # Check if file exists
    if not os.path.exists(input_path):
        click.echo(f"Error: File '{input_path}' not found.")
        return

    # Read data from CSV file
    try:
        click.echo(f"Reading data from {input_path}...")
        df = pd.read_csv(input_path)
        click.echo(f"Successfully loaded data with {len(df)} rows.")

        # Ad hoc filtering for specific models
        # List of models to exclude from the plot
        models_to_exclude = [
            # "Llama 3.1 70B",
            # "Llama 3.3 70B",
            # "Qwen 2.5 72B",
        ]

        # Filter out the specified models
        original_count = len(df)
        df = df[~df["Model"].isin(models_to_exclude)]
        filtered_count = original_count - len(df)

        if filtered_count > 0:
            click.echo(f"Filtered out {filtered_count} rows for excluded models: {', '.join(models_to_exclude)}")
    except Exception as e:
        click.echo(f"Error reading CSV file: {e}")
        return

    # Display column names for debugging
    click.echo("Columns in the CSV file: " + ", ".join(df.columns.tolist()))

    # Make sure required columns exist
    required_columns = ["Model", "Average", "Class", "Date"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        click.echo(f"Error: Missing required columns: {', '.join(missing_columns)}")
        click.echo("Available columns: " + ", ".join(df.columns.tolist()))
        return

    # Filter out rows without Average values
    df = df[df["Average"].notna()]
    click.echo(f"After filtering rows with missing Average values: {len(df)} rows remaining.")

    if len(df) == 0:
        click.echo("No valid data points found after filtering.")
        return

    # Convert Average to numeric, handling any conversion errors
    df["Average"] = pd.to_numeric(df["Average"], errors="coerce")
    df = df[df["Average"].notna()]  # Remove rows where conversion failed
    click.echo(f"After converting to numeric: {len(df)} valid rows.")

    # Convert Date to datetime
    df["DateTime"] = df["Date"].apply(parse_date)

    # Check for any date parsing failures
    if df["DateTime"].isna().any():
        click.echo("Warning: Some dates could not be parsed correctly.")
        for _, row in df[df["DateTime"].isna()][["Model", "Date"]].iterrows():
            click.echo(f"  {row['Model']}: {row['Date']}")
        # Remove rows with invalid dates
        df = df[df["DateTime"].notna()]

    # Sort by date
    df = df.sort_values("DateTime")

    # Print data summary
    click.echo("\nData Summary:")
    click.echo(f"Total models: {len(df)}")
    click.echo("Models per class:")
    for class_name, count in df["Class"].value_counts().items():
        click.echo(f"  {class_name}: {count}")

    # Define colors, markers, and styling for each class
    class_colors = {
        "Closed API model": "#B11BE8",  # dark purple
        "Open Weights model, 24-32B": "#255457",  # dark green
        "Open Weights model, 70B+": "#6FE0BA",  # light green
        "Fully Open Models": "#F0529C",  # dark pink
    }

    # Define text colors for annotations
    class_text_colors = {
        "Closed API model": "#B11BE8",  # dark purple
        "Open Weights model, 24-32B": "#255457",  # dark green
        "Open Weights model, 70B+": "#255457",  # darker green
        "Fully Open Models": "#a51c5c",  # darker pink
    }

    # Define marker styles for each class
    class_markers = {
        "Closed API model": "X",  # Using uppercase X for better visibility
        "Open Weights model, 24-32B": "D",  # diamond
        "Open Weights model, 70B+": "s",  # square
        "Fully Open Models": "*",  # star
    }

    # UPDATED: Increase marker sizes significantly
    class_marker_sizes = {
        "Closed API model": 300,  # Increased from 150
        "Open Weights model, 24-32B": 200,  # Increased from 100
        "Open Weights model, 70B+": 200,  # Increased from 100
        "Fully Open Models": 500,  # Increased from 350
    }

    # Define transparency levels for each class
    class_alpha = {
        "Closed API model": 0.7,
        "Open Weights model, 24-32B": 0.8,
        "Open Weights model, 70B+": 0.9,
        "Fully Open Models": 1.0,
    }

    # Model name cleanup for better display
    model_name_replacements = {
        "OLMo 2 7B": "OLMo-2-7B",
        "OLMo 2 13B": "OLMo-2-13B",
        "OLMo 2 32B": "OLMo-2-32B",
        "GPT-3.5 Turbo 0125": "GPT-3.5",
    }

    # Apply model name replacements
    df["DisplayModel"] = df["Model"].replace(model_name_replacements)

    # Add colors for any classes not in the predefined mapping
    for class_name in df["Class"].unique():
        if class_name not in class_colors:
            # Generate a random color for any class not in our predefined mapping
            random_color = "#{:06x}".format(np.random.randint(0, 0xFFFFFF))
            class_colors[class_name] = random_color
            class_text_colors[class_name] = random_color
            class_markers[class_name] = "o"  # default marker
            class_marker_sizes[class_name] = 160  # UPDATED: Increased default size
            class_alpha[class_name] = 0.8  # default alpha
            click.echo(f"Assigned random color {random_color} to class {class_name}")

    # UPDATED: Increase figure size
    plt.figure(figsize=(16, 14))  # Increased from (12, 10)

    # Set a higher z-order for the scatter plots to ensure they're above the grid
    for class_name, group in df.groupby("Class"):
        plt.scatter(
            group["DateTime"],
            group["Average"],
            color=class_colors.get(class_name, "#999999"),
            label=class_name,
            s=class_marker_sizes.get(class_name, 200),  # UPDATED: Increased default
            alpha=class_alpha.get(class_name, 1.0),
            marker=class_markers.get(class_name, "o"),
            edgecolors="white",
            linewidths=1.5,  # UPDATED: Increased from 0.8
            zorder=3,
        )

        # Add model name labels with custom styling
        for _, row in group.iterrows():
            # UPDATED: Removed bounding box and increased font size
            plt.annotate(
                row["DisplayModel"],
                (row["DateTime"], row["Average"]),
                xytext=(0, 18),  # UPDATED: Increased offset from 15
                textcoords="offset points",
                ha="center",
                fontsize=16,  # UPDATED: Increased from 12
                fontfamily="Manrope",
                fontweight="bold",  # UPDATED: Changed to bold
                color=class_text_colors.get(class_name, "black"),
                # Removed the bbox parameter to eliminate boxes
            )

    # Adjust y-axis range with some padding
    ymin = df["Average"].min() - 20  # Add padding below
    ymax = df["Average"].max() + 10  # Add padding above
    plt.ylim(ymin, ymax)

    # Add subtle background grid - vertical only
    plt.grid(True, axis="x", linestyle="--", alpha=0.3, color="#cccccc", zorder=1)

    # UPDATED: Style the axis labels and title with larger font sizes
    plt.xlabel("Date (MM/YY)", fontsize=18, fontfamily="Manrope", fontweight="bold", color="#333333")
    plt.ylabel(
        "Average Performance (10 benchmarks)",
        fontsize=18,
        fontfamily="Manrope",
        fontweight="bold",
        color="#333333",
    )

    # Format x-axis with MM/YY dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))

    # Use a more frequent month locator to compress the x-axis visually
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    # Set x-axis limits to compress the timeline
    min_date = df["DateTime"].min()
    max_date = df["DateTime"].max()

    # Add some padding to avoid points at the edges
    date_range = max_date - min_date
    padding = date_range * 0.05  # 5% padding on each side

    plt.xlim(min_date - padding, max_date + padding)

    # UPDATED: Rotate x-axis labels and style them with larger font size
    plt.xticks(rotation=45, ha="right", fontfamily="Manrope", fontweight="bold", fontsize=15, color="#333333")
    plt.yticks(fontfamily="Manrope", fontweight="bold", fontsize=15, color="#333333")

    # Style the spines (borders)
    for spine in plt.gca().spines.values():
        spine.set_color("#dddddd")
        spine.set_linewidth(1.2)  # UPDATED: Increased from 0.8

    # UPDATED: Style the tick marks with larger size
    plt.tick_params(axis="both", which="major", width=1.2, length=6, colors="#555555", pad=6)

    # Create custom legend handles
    custom_handles = []
    custom_labels = []

    # Get unique classes and sort them for consistent legend order
    unique_classes = sorted(df["Class"].unique())

    # Create custom Line2D objects for the legend
    for class_name in unique_classes:
        # UPDATED: Create a custom Line2D object with much larger marker size
        handle = Line2D(
            [],
            [],
            marker=class_markers.get(class_name, "o"),
            markersize=class_marker_sizes.get(class_name, 200) / 8
            if class_name != "Fully Open Models"
            else 30,  # UPDATED: Increased sizes
            color=class_colors.get(class_name, "#999999"),
            markeredgecolor="white",
            markeredgewidth=1.5,  # UPDATED: Increased from 1.0
            linestyle="None",
        )

        custom_handles.append(handle)
        custom_labels.append(class_name)

    # UPDATED: Create a new legend with our custom handles and larger font size
    legend = plt.legend(
        custom_handles,
        custom_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(custom_labels),
        frameon=True,
        fontsize=12,  # UPDATED: Increased from 11
        framealpha=0.9,
        edgecolor="#dddddd",
        handletextpad=0.8,  # UPDATED: Increased space between marker and text
    )

    # Apply custom styling to legend
    legend.get_frame().set_linewidth(1.2)  # UPDATED: Increased from 0.8
    plt.setp(legend.get_texts(), fontfamily="Manrope", fontweight="bold", color="#333333")

    # Set the figure background color for a cleaner look
    fig = plt.gcf()
    fig.patch.set_facecolor("#ffffff")  # White background
    plt.gca().set_facecolor("#ffffff")  # White plot area

    # Adjust layout to reduce whitespace
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Determine output file name based on input file if not provided
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_scatter_plot.png"

    # Save the figure with higher DPI for better quality
    plt.savefig(output_path, dpi=400, bbox_inches="tight")  # UPDATED: Increased DPI from 300
    click.echo(f"\nPlot saved to {output_path}")

    click.echo("Done!")


if __name__ == "__main__":
    main()
