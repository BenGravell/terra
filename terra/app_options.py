from dataclasses import dataclass, field


EPS = 1e-3  # A small constant

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

    def __post_init__(self):
        self.overall_group_name = "Overall"
        self.overall_fields = ["cf_score_weight", "ql_score_weight"]
        self.ql_group_name = "Quality-of-Life"
        self.ql_fields = ["hp_score_weight", "sp_score_weight", "hf_score_weight"]

    def are_weights_valid(self, fields: list[str], group_name: str):
        valid = any([getattr(self, field) > EPS for field in fields])
        message = (
            f"The {group_name} Preference weights are all zero - the {group_name} Score is not well-defined! Please set"
            " at least one weight greater than zero."
            if not valid
            else ""
        )
        return valid, message

    def are_weights_valid_100(self, fields: list[str], group_name: str):
        pct_sum = sum([round(100 * getattr(self, field)) for field in fields])
        valid = pct_sum == 100
        message = (
            f"The {group_name} Preference weights do not add up to 100 (they add up to {pct_sum} right now) - the"
            f" {group_name} Score is not well-defined! Please make sure the weights add up to 100."
            if not valid
            else ""
        )
        return valid, message

    @property
    def are_overall_weights_valid(self):
        return self.are_weights_valid(fields=self.overall_fields, group_name=self.overall_group_name)

    @property
    def are_overall_weights_valid_100(self):
        return self.are_weights_valid_100(fields=self.overall_fields, group_name=self.overall_group_name)

    @property
    def are_ql_weights_valid(self):
        return self.are_weights_valid(fields=self.ql_fields, group_name=self.ql_group_name)

    @property
    def are_ql_weights_valid_100(self):
        return self.are_weights_valid_100(fields=self.ql_fields, group_name=self.ql_group_name)

    @property
    def are_all_options_valid(self):
        all_valid = True
        messages = []
        props = [
            self.are_overall_weights_valid,
            self.are_overall_weights_valid_100,
            self.are_ql_weights_valid,
            self.are_ql_weights_valid_100,
        ]
        for prop in props:
            valid, message = prop
            if not valid:
                all_valid = False
                messages.append(message)
        return all_valid, messages

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
