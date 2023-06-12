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

EPS = 1e-6

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

    # Weights
    cf_score_weight: float = 1.0
    pf_score_weight: float = 0.0
    ef_score_weight: float = 0.0

    # Results options
    N: int = 5
    show_radar: bool = False
    field_for_world_map: str = "overall_score"
    world_map_projection_type: str = "kavrayskiy7"

    @property
    def do_filter_culture_fit(self):
        return not self.cf_score_min < EPS
    
    @property
    def do_filter_freedom(self):
        return not (self.pf_score_min < EPS and self.ef_score_min < EPS)

    @property
    def do_filter_english(self):
        return not self.english_ratio_min < EPS
