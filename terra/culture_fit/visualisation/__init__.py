import math
from typing import Optional

import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet, ImageURL
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from terra.data_handling.loading import COUNTRY_FLAG_IMAGE_URLS


DEFAULT_COLORMAP = "coolwarm_r"
DEFAULT_FORMAT = ".0f"
DEFAULT_TEXT_ROTATION_DEGREES = 80
POLAR_PLOT_Y_TICKS_RANGE = [i for i in range(0, 120, 20)]
POLAR_Y_TICKS_SIZE = 7
POLAR_X_TICKS_SIZE = 8
COLOR_GREY = "grey"
COLOR_DARKS_LATE_GREY = "darkslategrey"
FONT_FAMILY_SANS_SERIF = "sans-serif"
MAX_VALUE_PER_DIMENSION = 100
SOLID_LINE_STYLE = "solid"
RADAR_PLOTS_COLOR_MAP = "Set3"
RADAR_PLOTS_COLOR_MAP_N = 12  # number of unique colors in RADAR_PLOTS_COLOR_MAP
RADAR_PLOTS_PADDING = 1.2
RADAR_PLOT_SIZE = 1000
DISPLAY_DPI = 100
RADAR_PLOT_TITLE_FONT_SIZE = 11
RADAR_PLOT_TITLE_Y_POSITION = 1.2
RADAR_PLOT_ALPHA_CHANNEL = 0.4
TEXT_COORDS_OFFSET_POINTS = "offset points"
MIN_SIDE_LEN_SPIDER_PLOTS = 5


def generate_heatmap(distances: pd.DataFrame, show_clusters: bool) -> plt.Figure:
    fig, ax = plt.subplots()
    plt.xticks(rotation=DEFAULT_TEXT_ROTATION_DEGREES)
    if show_clusters:
        fig = sns.clustermap(distances, cmap=DEFAULT_COLORMAP, annot=True, fmt=DEFAULT_FORMAT)
    else:
        sns.heatmap(distances, ax=ax, cmap=DEFAULT_COLORMAP, annot=True, fmt=DEFAULT_FORMAT)
        fig.tight_layout()
    return fig


def generate_scatterplot(
    coords: pd.DataFrame,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
) -> plt.Figure:
    data = {str(key): val for key, val in coords.to_dict(orient="list").items()}
    data["names"] = coords.index

    if x_column is None:
        x_column = coords.columns[0]
    if y_column is None:
        y_column = coords.columns[1]

    max_x, max_y = max(data[x_column]), max(data[y_column])
    min_x, min_y = min(data[x_column]), min(data[y_column])
    dx = max_x - min_x
    dy = max_y - min_y
    fig = figure(
        width=600,
        height=600,
        x_axis_label=x_column,
        y_axis_label=y_column,
        match_aspect=True,
    )
    source = ColumnDataSource(data=data)
    labels = LabelSet(
        x=x_column,
        y=y_column,
        text="names",
        x_offset=10,
        y_offset=10,
        source=source,
        render_mode="canvas",
    )
    fig.add_layout(labels)
    for country_name, country_coords in coords.iterrows():
        img = ImageURL(
            url=dict(value=COUNTRY_FLAG_IMAGE_URLS[country_name]),
            x=country_coords[x_column],
            y=country_coords[y_column],
            w=0.05 * dx,
            h=0.02 * dy,
            anchor="center",
        )
        fig.add_glyph(source, img)
    return fig


def generate_radar_plot(
    dimensions: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> plt.Figure:
    fig = plt.figure(
        figsize=(RADAR_PLOT_SIZE / DISPLAY_DPI, RADAR_PLOT_SIZE / DISPLAY_DPI),
        dpi=DISPLAY_DPI,
    )

    # Create a color palette:
    my_palette = plt.cm.get_cmap(RADAR_PLOTS_COLOR_MAP, RADAR_PLOTS_COLOR_MAP_N)

    # Loop to plot
    for idx, country in enumerate(dimensions.columns):
        make_spider(
            idx,
            country,
            my_palette(idx % RADAR_PLOTS_COLOR_MAP_N),
            dimensions,
            reference,
        )

    fig.tight_layout(pad=RADAR_PLOTS_PADDING)
    return fig


def generate_choropleth(dimensions: pd.DataFrame, dimension_name: str) -> plt.Figure:
    transposed = dimensions.transpose()
    transposed.reset_index(inplace=True)
    transposed = transposed.rename(columns={"index": "country"})
    fig = px.choropleth(
        transposed,
        locationmode="country names",
        locations="country",
        color=dimension_name,
        hover_name="country",
        color_continuous_scale=px.colors.sequential.Plasma,
    )
    return fig


def make_spider(
    col: int,
    title: str,
    color: str,
    dimensions: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> None:
    # number of variable
    categories = list(dimensions.index)
    num_categories = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(num_categories) * 2 * math.pi for n in range(num_categories)]
    angles += angles[:1]

    # Initialise the spider plot
    side_len = max(math.ceil(math.sqrt(len(dimensions.columns))), MIN_SIDE_LEN_SPIDER_PLOTS)
    ax = plt.subplot(side_len, side_len, col + 1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color=COLOR_GREY, size=POLAR_X_TICKS_SIZE)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(
        POLAR_PLOT_Y_TICKS_RANGE,
        list(map(str, POLAR_PLOT_Y_TICKS_RANGE)),
        color=COLOR_GREY,
        size=POLAR_Y_TICKS_SIZE,
    )
    plt.ylim(0, MAX_VALUE_PER_DIMENSION)

    # Ind1
    values = dimensions[title].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle=SOLID_LINE_STYLE)
    ax.fill(angles, values, color=color, alpha=RADAR_PLOT_ALPHA_CHANNEL)

    # Ind2
    if reference is not None:
        values = reference["user"].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color="red", linewidth=1, linestyle="dashed")
        ax.fill(angles, values, color="red", alpha=0.1)

    # Add a title
    plt.title(
        title,
        size=RADAR_PLOT_TITLE_FONT_SIZE,
        color=color,
        y=RADAR_PLOT_TITLE_Y_POSITION,
    )
