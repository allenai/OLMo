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
        "Closed API model": "x",  # x
        "Open Weights model, 24-32B": "D",  # diamond
        "Open Weights model, 70B+": "s",  # square
        "Fully Open Models": "*",  # star
    }

    # Define marker sizes for each class
    class_marker_sizes = {
        "Closed API model": 80,
        "Open Weights model, 24-32B": 100,
        "Open Weights model, 70B+": 120,
        "Fully Open Models": 150,
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
            class_marker_sizes[class_name] = 80  # default size
            class_alpha[class_name] = 0.8  # default alpha
            click.echo(f"Assigned random color {random_color} to class {class_name}")

    # Create the plot
    plt.figure(figsize=(12, 10))  # More square dimensions

    # Set a higher z-order for the scatter plots to ensure they're above the grid
    for class_name, group in df.groupby("Class"):
        plt.scatter(
            group["DateTime"],
            group["Average"],
            color=class_colors.get(class_name, "#999999"),
            label=class_name,
            s=class_marker_sizes.get(class_name, 100),  # Get custom size or default
            alpha=class_alpha.get(class_name, 0.8),  # Get custom alpha or default
            marker=class_markers.get(class_name, "o"),  # Get custom marker or default
            edgecolors="white",  # Add white edge for better visibility
            linewidths=0.8,  # Width of marker edge
            zorder=3,
        )

        # Add model name labels with custom styling
        for _, row in group.iterrows():
            plt.annotate(
                row["DisplayModel"],  # Use the cleaned model name
                (row["DateTime"], row["Average"]),
                xytext=(0, 10),  # Offset text by 10 points above
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontfamily="Manrope",
                fontweight="medium",
                color=class_text_colors.get(class_name, "black"),  # Use class-specific text color
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec=class_colors.get(class_name, "gray"),  # Use class color for border
                    alpha=0.9,
                ),
            )

    # Adjust y-axis range with some padding
    ymin = df["Average"].min() - 2  # Add padding below
    ymax = df["Average"].max() + 2  # Add padding above
    plt.ylim(ymin, ymax)

    # Add subtle background grid - vertical only
    plt.grid(True, axis="x", linestyle="--", alpha=0.3, color="#cccccc", zorder=1)

    # Style the axis labels and title
    plt.xlabel("Date (MM/YY)", fontsize=14, fontfamily="Manrope", fontweight="medium", color="#333333")
    plt.ylabel(
        "Average Performance (10 benchmarks)",
        fontsize=14,
        fontfamily="Manrope",
        fontweight="medium",
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

    # Rotate x-axis labels and style them
    plt.xticks(rotation=45, ha="right", fontfamily="Manrope", fontweight="medium", fontsize=11, color="#333333")
    plt.yticks(fontfamily="Manrope", fontweight="medium", fontsize=11, color="#333333")

    # Style the spines (borders)
    for spine in plt.gca().spines.values():
        spine.set_color("#dddddd")  # Lighter border
        spine.set_linewidth(0.8)  # Thinner border

    # Style the tick marks
    plt.tick_params(axis="both", which="major", width=0.8, length=4, colors="#555555", pad=4)

    # Add legend with custom styling
    legend = plt.legend(
        # title="Model Class",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(df["Class"].unique()),
        frameon=True,
        fontsize=11,
        framealpha=0.9,
        edgecolor="#dddddd",
    )

    # Apply custom styling to legend
    legend.get_frame().set_linewidth(0.8)  # Thinner border
    plt.setp(legend.get_title(), fontfamily="Manrope", fontweight="bold", fontsize=13, color="#333333")

    # Style each legend handle according to its class
    for i, (text, handle) in enumerate(zip(legend.get_texts(), legend.legend_handles)):
        class_name = text.get_text()

        # Set text style
        plt.setp(text, fontfamily="Manrope", fontweight="medium", color="#333333")

        # For PathCollection objects (which scatter plots create), we need to handle differently
        # than regular Line2D objects
        if class_name in class_markers:
            # For scatter plot legend handles (PathCollection objects)
            if hasattr(handle, "set_paths"):
                # Get the path for the marker we want
                from matplotlib.markers import MarkerStyle

                marker = MarkerStyle(class_markers[class_name])
                path = marker.get_path().transformed(marker.get_transform())
                handle.set_paths([path])

            # If it's a regular Line2D object (not likely with scatter, but just in case)
            elif hasattr(handle, "set_marker"):
                handle.set_marker(class_markers[class_name])

            # Set other properties that should work for both types
            handle.set_alpha(1.0)  # Always full opacity in legend

            # Set edge color if the method exists
            if hasattr(handle, "set_edgecolor"):
                handle.set_edgecolor("white")
                handle.set_linewidth(0.8)

    # Set the figure background color for a cleaner look
    fig = plt.gcf()
    fig.patch.set_facecolor("#ffffff")  # White background
    plt.gca().set_facecolor("#ffffff")  # White plot area

    # Add a subtle text annotation for data source
    plt.figtext(
        0.02, 0.02, "Source: Model Evaluation Data", fontsize=8, fontfamily="Manrope", color="#888888", ha="left"
    )

    # Adjust layout to make room for labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # Leave room for title and footer

    # Determine output file name based on input file if not provided
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_scatter_plot.png"

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    click.echo(f"\nPlot saved to {output_path}")

    # Show the plot
    plt.show()

    click.echo("Done!")


if __name__ == "__main__":
    main()
