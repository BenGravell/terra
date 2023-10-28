import dataclasses
from functools import partial
from typing import Any

from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.decomposition import (
    PCA,
    FastICA,
    NMF,
    MiniBatchSparsePCA,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
import pandas as pd
import plotly.express as px
from plotly import figure_factory as ff
import streamlit as st


from terra.data_handling.utils import df_format_func
from terra import app_options as ao
from terra import app_config
from terra import color_options
from terra.ui.sections import UISection
from terra.data_handling.processing import FieldListBank


DECOMPOSER_MAP = {
    "PCA": PCA,
    "FastICA": FastICA,
    "NMF": NMF,
    "MiniBatchSparsePCA": MiniBatchSparsePCA,
    "SparsePCA": SparsePCA,
    "TruncatedSVD": TruncatedSVD,
    "t-SNE": TSNE,
    "UMAP": UMAP,
}


PDIST_METRIC_OPTIONS = [
    "cityblock",
    "euclidean",
    "cosine",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

LINKAGE_METHOD_OPTIONS = [
    "complete",
    "average",
    "single",
    "weighted",
    "centroid",
    "median",
    "ward",
]


@dataclasses.dataclass
class FieldOptions:
    dr_fields: list[str]


@dataclasses.dataclass
class DimensionalityReductionOptions:
    decomposer_name: Any
    decomposer_kwargs: dict[str, Any]


@dataclasses.dataclass
class HdbscanClusteringOptions:
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float


@dataclasses.dataclass
class HierarchicalClusteringOptions:
    distance_metric: str
    linkage_method: str


@dataclasses.dataclass
class ClusteringOptions:
    clustering_method_name: str
    clusterer_options: HdbscanClusteringOptions | HierarchicalClusteringOptions
    cluster_in_projected_space: bool


@dataclasses.dataclass
class HdbscanClusteringPlotOptions:
    show_country_text: bool
    marker_size_field: str
    marker_size_power: float


@dataclasses.dataclass
class HierarchicalClusteringPlotOptions:
    color_threshold: float
    orientation: str


@dataclasses.dataclass
class ClusteringPlotOptions:
    clustering_method_name: str
    clusterer_plot_options: HdbscanClusteringPlotOptions | HierarchicalClusteringPlotOptions


# TODO use session_state_manager for this
def set_dr_fields_callback(fields):
    st.session_state.dr_fields = fields


@st.cache_data(ttl=app_config.TTL)
def get_dimension_reduced_df(df, options: DimensionalityReductionOptions):
    decomposer_class = DECOMPOSER_MAP[options.decomposer_name]
    decomposer = decomposer_class(**options.decomposer_kwargs)

    projection = decomposer.fit_transform(df)
    df_projection = pd.DataFrame(projection).rename(columns={0: "x", 1: "y"})
    df_projection.index = df.index
    return df_projection


@dataclasses.dataclass
class DimensionalityReductionClusteringOptions:
    field_options: FieldOptions
    dimensionality_reduction_options: DimensionalityReductionOptions
    clustering_options: ClusteringOptions
    clustering_plot_options: ClusteringPlotOptions


class FieldOptionsSection(UISection):
    def run(self, flb: FieldListBank):
        dr_fields = st.multiselect(
            "Fields for Dimensionality Reduction & Clustering",
            options=flb.numeric_plottable_fields,
            format_func=df_format_func,
            key="dr_fields",
        )

        cols = st.columns(4)
        with cols[0]:
            st.button(
                "Set Fields to All",
                use_container_width=True,
                on_click=set_dr_fields_callback,
                args=[flb.numeric_plottable_fields],
            )
        with cols[1]:
            st.button(
                "Set Fields to Culture Dimensions",
                use_container_width=True,
                on_click=set_dr_fields_callback,
                args=[flb.culture_fields],
            )
        with cols[2]:
            st.button(
                "Set Fields to Quality-of-Life Dimensions",
                use_container_width=True,
                on_click=set_dr_fields_callback,
                args=[flb.quality_of_life_fields],
            )
        with cols[3]:
            st.button(
                "Set Fields to Climate Dimensions",
                use_container_width=True,
                on_click=set_dr_fields_callback,
                args=[flb.climate_fields],
            )

        return FieldOptions(dr_fields)


class DimensionalityReductionOptionsSection(UISection):
    def run(self, flb):
        decomposer_name = st.selectbox(
            "Dimensionality Reduction Method",
            options=[
                "UMAP",
                "t-SNE",
                "PCA",
                "SparsePCA",
                "TruncatedSVD",
                "FastICA",
                "NMF",
            ],
        )
        with st.form("dimesionality_reduction_options"):
            decomposer_kwargs = {}

            if decomposer_name in [
                "PCA",
                "SparsePCA",
                "TruncatedSVD",
                "FastICA",
                "NMF",
            ]:
                cols = st.columns(0 + 1)

            elif decomposer_name == "t-SNE":
                cols = st.columns(2 + 1)
                with cols[0]:
                    perplexity = st.slider(
                        "Perplexity",
                        min_value=1.0,
                        max_value=30.0,
                        value=10.0,
                        help=(
                            "The perplexity is related to the number of nearest neighbors that is used in other"
                            " manifold learning algorithms. Larger datasets usually require a larger perplexity."
                            " Different values can result in significantly different results. The perplexity must"
                            " be less than the number of samples. See"
                            " https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"
                        ),
                    )
                with cols[1]:
                    early_exaggeration = st.slider(
                        "Early Exaggeration",
                        min_value=1.0,
                        max_value=30.0,
                        value=10.0,
                        help=(
                            "Controls how tight natural clusters in the original space are in the embedded space"
                            " and how much space will be between them. For larger values, the space between natural"
                            " clusters will be larger in the embedded space. Again, the choice of this parameter is"
                            " not very critical. If the cost function increases during initial optimization, the"
                            " early exaggeration factor or the learning rate might be too high. See"
                            " https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"
                        ),
                    )

                decomposer_kwargs["perplexity"] = perplexity
                decomposer_kwargs["early_exaggeration"] = early_exaggeration

            elif decomposer_name == "UMAP":
                cols = st.columns(2 + 1)
                with cols[0]:
                    n_neighbors = st.slider(
                        "Number of Neighbors",
                        min_value=1,
                        max_value=50,
                        value=15,
                        help=(
                            "This parameter controls how UMAP balances local versus global structure in the data."
                            " It does this by constraining the size of the local neighborhood UMAP will look at"
                            " when attempting to learn the manifold structure of the data. This means that low"
                            " values will force UMAP to concentrate on very local structure (potentially to the"
                            " detriment of the big picture), while large values will push UMAP to look at larger"
                            " neighborhoods of each point when estimating the manifold structure of the data,"
                            " losing fine detail structure for the sake of getting the broader of the data. See"
                            " https://umap-learn.readthedocs.io/en/latest/parameters.html"
                        ),
                    )
                with cols[1]:
                    min_dist = st.slider(
                        "Minimum Distance in Projected Space",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        help=(
                            "This parameter controls how tightly UMAP is allowed to pack points together. It, quite"
                            " literally, provides the minimum distance apart that points are allowed to be in the"
                            " low dimensional representation. This means that low values will result in clumpier"
                            " embeddings. This can be useful if you are interested in clustering, or in finer"
                            " topological structure. Larger values will prevent UMAP from packing points together"
                            " and will focus on the preservation of the broad topological structure instead. See"
                            " https://umap-learn.readthedocs.io/en/latest/parameters.html"
                        ),
                    )

                decomposer_kwargs["n_neighbors"] = n_neighbors
                decomposer_kwargs["min_dist"] = min_dist

            with cols[-1]:
                random_state = st.number_input("Random State", min_value=0, max_value=10, value=0, step=1)
                decomposer_kwargs["random_state"] = random_state

            st.form_submit_button("Update Dimensionality Reduction Options")

        return DimensionalityReductionOptions(decomposer_name, decomposer_kwargs)


class ClusteringOptionsSection(UISection):
    def run(self):
        clustering_method_name = st.selectbox("Clustering Method", options=["HDBSCAN", "Hierarchical"])

        with st.form("clustering_options"):
            if clustering_method_name == "HDBSCAN":
                cols = st.columns(3)
                with cols[0]:
                    min_cluster_size = st.slider(
                        "Min Cluster Size",
                        min_value=2,
                        max_value=20,
                        step=1,
                        value=3,
                        help=(
                            "The smallest size grouping that you wish to consider a cluster. See"
                            " https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size."
                        ),
                    )
                with cols[1]:
                    min_samples = st.slider(
                        "Min Samples",
                        min_value=1,
                        max_value=20,
                        step=1,
                        value=2,
                        help=(
                            "How conservative you want you clustering to be. The larger the value you provide, the"
                            " more conservative the clustering - more points will be declared as noise, and"
                            " clusters will be restricted to progressively more dense areas. See"
                            " https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size."
                        ),
                    )
                with cols[2]:
                    cluster_selection_epsilon = st.slider(
                        "Cluster Selection Epsilon",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        help=(
                            "Ensures that clusters below the given threshold are not split up any further. See"
                            " https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-cluster-selection-epsilon."
                        ),
                    )
                clusterer_options = HdbscanClusteringOptions(min_cluster_size, min_samples, cluster_selection_epsilon)

            elif clustering_method_name == "Hierarchical":
                cols = st.columns(2)
                with cols[0]:
                    distance_metric = st.selectbox(
                        "Distance Metric",
                        options=PDIST_METRIC_OPTIONS,
                        help=(
                            "See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html"
                        ),
                    )
                with cols[1]:
                    linkage_method = st.selectbox(
                        "Linkage Method",
                        options=LINKAGE_METHOD_OPTIONS,
                        help=(
                            "See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html"
                        ),
                    )
                clusterer_options = HierarchicalClusteringOptions(distance_metric, linkage_method)

            cluster_in_projected_space = st.toggle(
                "Cluster in Projected Space",
                value=False,
                help=(
                    "If False, clustering will occur in the original pre-dimension-reduced space. If True,"
                    " clustering will occur in the projected post-dimension-reduced space. Because t-SNE and"
                    " UMAP do not preserve distances, clustering in the projected space is not as"
                    " meaningful/principled, but can give more intuitive clusterings when viewed on the plot."
                ),
            )

            st.form_submit_button("Update Clustering Options")

        return ClusteringOptions(
            clustering_method_name,
            clusterer_options,
            cluster_in_projected_space,
        )


class ClusteringPlotOptionsSection(UISection):
    def run(self, flb, clustering_options: ClusteringOptions):
        with st.form("dim_red_cluster_plot_options"):
            if clustering_options.clustering_method_name == "HDBSCAN":
                cols = st.columns(3)
                with cols[0]:
                    show_country_text = st.checkbox("Show Country Name Text on Plot", value=True)
                with cols[1]:
                    marker_size_field = st.selectbox(
                        "Marker Size Field",
                        options=flb.plottable_fields,
                        format_func=df_format_func,
                    )
                with cols[2]:
                    marker_size_power = st.slider(
                        "Marker Size Power",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5,
                        help=(
                            "Power to which to raise the field's value. Higher powers will exaggerate differences"
                            " between points, while lower values will diminish them. A power of 1 will make the marker"
                            " size linearly proportional to the field value. A power of 0 will make all points the same"
                            " size, regardless of the field value."
                        ),
                    )
                clusterer_plot_options = HdbscanClusteringPlotOptions(
                    show_country_text, marker_size_field, marker_size_power
                )

            elif clustering_options.clustering_method_name == "Hierarchical":
                cols = st.columns(2)
                with cols[0]:
                    color_threshold = st.slider(
                        "Cluster Color Threshold",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.5,
                        help="Lower values will result in more clusters.",
                    )
                with cols[1]:
                    orientation = st.selectbox(
                        "Plot Orientation",
                        options=["bottom", "top", "right", "left"],
                    )
                clusterer_plot_options = HierarchicalClusteringPlotOptions(color_threshold, orientation)

            st.form_submit_button("Update Clustering Plot Options")

        return ClusteringPlotOptions(
            clustering_options.clustering_method_name,
            clusterer_plot_options,
        )


class DimensionalityReductionClusteringOptionsSection(UISection):
    def run(self, flb):
        tab_names = [
            "Field Options",
            "Dimensionality Reduction Options",
            "Clustering Options",
            "Clustering Plot Options",
        ]
        tabs = st.tabs(tab_names)

        with tabs[tab_names.index("Field Options")]:
            field_options = FieldOptionsSection().run(flb)

        with tabs[tab_names.index("Dimensionality Reduction Options")]:
            dimensionality_reduction_options = DimensionalityReductionOptionsSection().run(flb)

        with tabs[tab_names.index("Clustering Options")]:
            clustering_options = ClusteringOptionsSection().run()

        with tabs[tab_names.index("Clustering Plot Options")]:
            clustering_plot_options = ClusteringPlotOptionsSection().run(flb, clustering_options)

        return DimensionalityReductionClusteringOptions(
            field_options, dimensionality_reduction_options, clustering_options, clustering_plot_options
        )


class DimensionalityReductionClusteringSection(UISection):
    def run(self, df: pd.DataFrame, app_options: ao.AppOptions, num_total: int, selected_country: str):
        flb = FieldListBank(df)

        # Use containers to have the plot above the options, since the options will take up a lot of space
        plot_container = st.container()
        options_container = st.container()

        # Set default for multiselect
        if "dr_fields" not in st.session_state:
            set_dr_fields_callback(flb.culture_fields)

        with options_container:
            options = DimensionalityReductionClusteringOptionsSection().run(flb)

        # Data processing
        df_for_dr = df.set_index("country")[options.field_options.dr_fields].dropna()
        df_projection = get_dimension_reduced_df(
            df_for_dr,
            options.dimensionality_reduction_options,
        )

        if options.clustering_options.cluster_in_projected_space:
            df_for_clustering = df_projection
        else:
            df_for_clustering = df_for_dr

        if options.clustering_options.clustering_method_name == "HDBSCAN":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=options.clustering_options.clusterer_options.min_cluster_size,
                min_samples=options.clustering_options.clusterer_options.min_samples,
                cluster_selection_epsilon=options.clustering_options.clusterer_options.cluster_selection_epsilon,
            )
            clusterer.fit(df_for_clustering)
            df_clusters = pd.DataFrame(clusterer.labels_).rename(columns={0: "cluster"}).astype(str)
            df_clusters.index = df_for_clustering.index
            df_for_dr_plot = pd.concat([df.set_index("country"), df_projection, df_clusters], axis=1).reset_index()
            df_for_dr_plot["marker_size"] = (
                df_for_dr_plot[options.clustering_plot_options.clusterer_plot_options.marker_size_field]
                ** options.clustering_plot_options.clusterer_plot_options.marker_size_power
            )
            category_orders = {"cluster": [str(i) for i in range(-1, max(clusterer.labels_))]}
            scatter_kwargs = dict(
                x="x",
                y="y",
                hover_name="country_with_emoji",
                hover_data=["overall_score"],
                color="cluster",
                color_discrete_map=color_options.CLUSTER_COLOR_SEQUENCE_MAP,
                category_orders=category_orders,
                size="marker_size",
            )

            if options.clustering_plot_options.clusterer_plot_options.show_country_text:
                scatter_kwargs["text"] = "country"

            with plot_container:
                fig = px.scatter(df_for_dr_plot, **scatter_kwargs)
                if options.clustering_plot_options.clusterer_plot_options.show_country_text:
                    fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)

        elif options.clustering_options.clustering_method_name == "Hierarchical":
            distfun = partial(distance.pdist, metric=options.clustering_options.clusterer_options.distance_metric)
            linkagefun = partial(hierarchy.linkage, method=options.clustering_options.clusterer_options.linkage_method)
            fig = ff.create_dendrogram(
                df_for_clustering,
                orientation=options.clustering_plot_options.clusterer_plot_options.orientation,
                labels=df_for_clustering.index,
                distfun=distfun,
                linkagefun=linkagefun,
                color_threshold=options.clustering_plot_options.clusterer_plot_options.color_threshold,
            )
            fig.add_hline(
                y=options.clustering_plot_options.clusterer_plot_options.color_threshold,
                line_dash="dash",
                line_color="white",
                opacity=0.5,
            )

            with plot_container:
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    from terra import app_config
    from terra.data_handling.processing import process_data

    app_config.streamlit_setup()

    app_options = ao.AppOptions()
    df, num_total = process_data(app_options)
    selected_country = "United States"
    DimensionalityReductionClusteringSection().run(df, app_options, num_total, selected_country)
