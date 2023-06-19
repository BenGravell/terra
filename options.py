from dataclasses import dataclass


NONE_COUNTRY = "(none)"

PLOTLY_MAP_PROJECTION_TYPES = [
    "robinson",
    "orthographic",
    "kavrayskiy7",
    "winkel tripel",
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

    # Culture Fit
    culture_fit_preference_pdi: int = 50
    culture_fit_preference_idv: int = 50
    culture_fit_preference_mas: int = 50
    culture_fit_preference_uai: int = 50
    culture_fit_preference_lto: int = 50
    culture_fit_preference_ind: int = 50

    # Filters
    year_min: int = 2015
    year_max: int = 2020
    cf_score_min: float = 0.0
    hp_score_min: float = 0.0
    bn_score_min: float = 0.0
    fw_score_min: float = 0.0
    op_score_min: float = 0.0
    pf_score_min: float = 0.0
    ef_score_min: float = 0.0
    english_ratio_min: float = 0.0

    # Weights
    cf_score_weight: float = 1.0
    hp_score_weight: float = 0.0
    bn_score_weight: float = 0.0
    fw_score_weight: float = 0.0
    op_score_weight: float = 0.0
    pf_score_weight: float = 0.0
    ef_score_weight: float = 0.0

    @property
    def do_filter_culture_fit(self):
        return self.cf_score_min > EPS
    
    @property
    def do_filter_happy_planet(self):
        return self.hp_score_min > EPS

    @property
    def do_filter_social_progress(self):
        return any([val > EPS for val in [self.bn_score_min, self.fw_score_min, self.op_score_min]])

    @property
    def do_filter_freedom(self):
        return self.pf_score_min > EPS or self.ef_score_min > EPS

    @property
    def do_filter_english(self):
        return self.english_ratio_min > EPS
