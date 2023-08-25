# import matplotlib
import cmocean  # noqa

COLORMAP_OPTIONS = []
for cmap in cmocean.cm.cmapnames:
    COLORMAP_OPTIONS.extend([cmap, f"{cmap}_r"])
DEFAULT_COLORMAP = "deep_r"

# default_colormap = color_options.DEFAULT_COLORMAP
# default_colormap_not_reversed = copy(default_colormap)
# if default_colormap_not_reversed.endswith("r"):
#     default_colormap_not_reversed = default_colormap_not_reversed.split("_r")[0]

# plotly_colormaps = list(px.colors.named_colorscales())
# plotly_colormaps.remove(default_colormap_not_reversed)
# plotly_colormaps.insert(0, default_colormap_not_reversed)

CLUSTER_COLOR_SEQUENCE_MAP = {
    "-1": "#797979",
    "0": "#636EFA",
    "1": "#EF553B",
    "2": "#00CC96",
    "3": "#AB63FA",
    "4": "#FFA15A",
    "5": "#19D3F3",
    "6": "#FF6692",
    "7": "#B6E880",
    "8": "#FF97FF",
    "9": "#FECB52",
}
