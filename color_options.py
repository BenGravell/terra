import matplotlib
import cmocean  # noqa
import plotly.express as px


CHOROPLETH_COLORMAP = px.colors.sequential.deep_r

GLOBE_COLORMAP = matplotlib.cm.get_cmap("cmo.deep_r")

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
