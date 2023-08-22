from dataclasses import dataclass, field

TTL = 1 * 60 * 60  # Default time-to-live for st.cache_data, in seconds

NONE_COUNTRY = "(none)"

EPS = 1e-3  # a small constant

CONTINENT_OPTIONS = ["Africa", "Asia", "Europe", "Oceania", "North America", "South America"]


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
    cf_score_min: float = 0.0
    ql_score_min: float = 0.0
    hp_score_min: float = 0.0
    sp_score_min: float = 0.0
    hf_score_min: float = 0.0
    english_ratio_min: float = 0.0
    average_temperature_celsius_min: float = -10.0
    average_temperature_celsius_max: float = 30.0
    average_sunshine_hours_per_day_min: float = 3.0
    average_sunshine_hours_per_day_max: float = 10.0
    continents: list[str] = field(default_factory=lambda: CONTINENT_OPTIONS)

    # Weights
    cf_score_weight: float = 0.5
    ql_score_weight: float = 0.5
    hp_score_weight: float = 0.3
    sp_score_weight: float = 0.4
    hf_score_weight: float = 0.3

    @property
    def are_overall_weights_valid(self):
        overall_fields = ["cf_score_weight", "ql_score_weight"]
        if all([getattr(self, field) < EPS for field in overall_fields]):
            return False
        return True

    @property
    def are_overall_weights_valid_100(self):
        overall_fields = ["cf_score_weight", "ql_score_weight"]
        pct_sum = sum([round(100 * getattr(self, field)) for field in overall_fields])
        return pct_sum == 100, pct_sum

    @property
    def are_ql_weights_valid(self):
        ql_fields = ["hp_score_weight", "sp_score_weight", "hf_score_weight"]
        if all([getattr(self, field) < EPS for field in ql_fields]):
            return False

        return True

    @property
    def are_ql_weights_valid_100(self):
        ql_fields = ["hp_score_weight", "sp_score_weight", "hf_score_weight"]
        pct_sum = sum([round(100 * getattr(self, field)) for field in ql_fields])
        return pct_sum == 100, pct_sum

    @property
    def do_filter_culture_fit(self):
        return self.cf_score_min > EPS

    @property
    def do_filter_quality_of_life(self):
        return self.ql_score_min > EPS

    @property
    def do_filter_happy_planet(self):
        return self.hp_score_min > EPS

    @property
    def do_filter_social_progress(self):
        return self.sp_score_min > EPS

    @property
    def do_filter_freedom(self):
        return self.hf_score_min > EPS

    @property
    def do_filter_english(self):
        return self.english_ratio_min > EPS

    @property
    def do_filter_temperature(self):
        min_active = self.average_temperature_celsius_min > -10.0 + EPS
        max_active = self.average_temperature_celsius_max < 30.0 - EPS
        return min_active or max_active

    @property
    def do_filter_sunshine(self):
        min_active = self.average_sunshine_hours_per_day_min > 3.0 + EPS
        max_active = self.average_sunshine_hours_per_day_max < 10.0 - EPS
        return min_active or max_active

    @property
    def do_filter_continents(self):
        return any(continent not in self.continents for continent in CONTINENT_OPTIONS)
