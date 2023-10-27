import cmocean  # noqa


COLORMAP_OPTIONS = []
for cmap in cmocean.cm.cmapnames:
    COLORMAP_OPTIONS.extend([cmap, f"{cmap}_r"])
DEFAULT_COLORMAP = "deep_r"

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
