from dataclasses import dataclass


NONE_COUNTRY = "(none)"

PLOTLY_MAP_PROJECTION_TYPES = [
    "kavrayskiy7",
    "orthographic",
    "winkel tripel",
    "robinson",
    "equirectangular",
    "mercator",
    "natural earth",
    "miller",
    "eckert4",
    "azimuthal equal area",
    "azimuthal equidistant",
    "conic equal area",
    "conic conformal",
    "conic equidistant",
    "gnomonic",
    "stereographic",
    "mollweide",
    "hammer",
    "transverse mercator",
    "albers usa",
    "aitoff",
    "sinusoidal",
]


@dataclass
class AppOptions:
    """Class for app options."""

    # Time range
    year_min: int = 2015
    year_max: int = 2020

    # Culture fit
    culture_fit_preference_pdi: int = 50
    culture_fit_preference_idv: int = 50
    culture_fit_preference_mas: int = 50
    culture_fit_preference_uai: int = 50
    culture_fit_preference_lto: int = 50
    culture_fit_preference_ind: int = 50

    # Filters
    cf_score_min: float = 0.0
    pf_score_min: float = 0.0
    ef_score_min: float = 0.0
    english_ratio_min: float = 0.0

    do_filter_culture_fit: bool = False
    do_filter_freedom: bool = False
    do_filter_english: bool = False

    # Weights
    cf_score_weight: float = 1.0
    pf_score_weight: float = 0.0
    ef_score_weight: float = 0.0

    # Results options
    N: int = 5
    show_radar: bool = False
    field_for_world_map: str = "overall_score"
    world_map_projection_type: str = "kavrayskiy7"
