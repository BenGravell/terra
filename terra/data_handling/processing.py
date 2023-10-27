from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
import streamlit as st

from terra.culture_fit import country_data, dimensions_info, distance_calculations
from terra.data_handling.loading import DATA, DATA_DICT
from terra import app_options as ao
from terra import app_config


# TODO move to config file
class FieldListBank:
    """Class to hold various lists of fields."""

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.overall_fields = [
            "overall_score",
            "cf_score",
            "ql_score",
        ]
        self.quality_of_life_fields = [
            "hp_score",
            "sp_score",
            "hf_score",
        ]
        self.culture_fields = dimensions_info.DIMENSIONS
        self.climate_fields = ["average_temperature_celsius", "average_sunshine_hours_per_day"]
        self.geography_fields = ["continent"]

        self.plottable_fields = (
            self.overall_fields + self.quality_of_life_fields + self.culture_fields + self.geography_fields
        )

        # Special handling for optional fields
        self.optional_fields = [
            "english_ratio",
            "average_temperature_celsius",
            "average_sunshine_hours_per_day",
        ]
        for field in self.optional_fields:
            if df is not None and field in df.columns:
                self.plottable_fields += [field]

        self.numeric_plottable_fields = [x for x in self.plottable_fields if x != "continent"]

        self.plottable_field_default_index = self.plottable_fields.index("overall_score")
        self.numeric_plottable_field_default_index = self.numeric_plottable_fields.index("overall_score")


def get_user_ideal(app_options):
    user_ideal = country_data.types.CountryInfo(
        id=999,
        title="user",
        slug="user",
        pdi=app_options.culture_fit_preference_pdi,
        idv=app_options.culture_fit_preference_idv,
        mas=app_options.culture_fit_preference_mas,
        uai=app_options.culture_fit_preference_uai,
        lto=app_options.culture_fit_preference_lto,
        ind=app_options.culture_fit_preference_ind,
        adjective="user",
    )
    return user_ideal


class DataProcessor(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame, app_options: ao.AppOptions) -> pd.DataFrame:
        return


class SequentialProcessor(DataProcessor):
    def __init__(self, processors: list[DataProcessor]) -> None:
        self.processors = processors

    def process(self, df: pd.DataFrame, app_options: ao.AppOptions) -> pd.DataFrame:
        for processor in self.processors:
            df = processor.process(df, app_options)
        return df


class CultureFitProcessor(DataProcessor):
    def process(self, df: pd.DataFrame, app_options: ao.AppOptions) -> pd.DataFrame:
        user_ideal = get_user_ideal(app_options)

        all_countries = list(DATA_DICT["Culture Fit"].d.values())

        distances, max_distance = distance_calculations.compute_distances(
            countries_from=[user_ideal],
            countries_to=all_countries,
            distance_metric="Manhattan",
        )
        distances = distances.sort_values("user")
        distances = (
            distances / 100
        )  # Divide by the maximum possible deviation in each dimension, which is 100, to make this a unit distance.
        distances = distances / len(
            dimensions_info.DIMENSIONS
        )  # Divide by the number of dimensions to make this an average over dimensions.
        culture_fit_score = 1 - distances  # Define similarity as 1 - distance
        culture_fit_score = culture_fit_score.reset_index()
        culture_fit_score = culture_fit_score.rename(columns={"index": "country", "user": "cf_score"})

        df = df.merge(culture_fit_score, on="country")

        return df


class LanguagePrevalenceProcessor(DataProcessor):
    def process(self, df: pd.DataFrame, app_options: ao.AppOptions) -> pd.DataFrame:
        if app_options.do_filter_english:
            df = df.dropna(subset=["english_ratio"])
        return df


class OverallScoreProcessor(DataProcessor):
    def process(self, df: pd.DataFrame, app_options: ao.AppOptions) -> pd.DataFrame:
        # Make copies to protect app_options from modification

        # Quality-of-Life Score
        ql_subcodes = ["hp", "sp", "hf"]

        ql_weights = {f"{code}_score": deepcopy(getattr(app_options, f"{code}_score_weight")) for code in ql_subcodes}

        ql_weight_sum = sum([ql_weights[key] for key in ql_weights])
        for key in ql_weights:
            ql_weights[key] /= ql_weight_sum

        ql_score_names_to_weight = [f"{code}_score" for code in ql_subcodes]
        for score_name in ql_score_names_to_weight:
            df[f"{score_name}_weighted"] = df[score_name] * ql_weights[score_name]

        df["ql_score"] = 0.0
        for score_name in ql_score_names_to_weight:
            df["ql_score"] += df[f"{score_name}_weighted"]

        # Overall Score
        overall_subcodes = ["cf", "ql"]

        overall_weights = {
            f"{code}_score": deepcopy(getattr(app_options, f"{code}_score_weight")) for code in overall_subcodes
        }

        overall_weight_sum = sum([overall_weights[key] for key in overall_weights])
        for key in overall_weights:
            overall_weights[key] /= overall_weight_sum

        overall_score_names_to_weight = [f"{code}_score" for code in overall_subcodes]
        for score_name in overall_score_names_to_weight:
            df[f"{score_name}_weighted"] = df[score_name] * overall_weights[score_name]

        df["overall_score"] = 0.0
        for score_name in overall_score_names_to_weight:
            df["overall_score"] += df[f"{score_name}_weighted"]

        df = df.sort_values("overall_score", ascending=False)

        return df


class RanksProcessor(DataProcessor):
    def process(self, df: pd.DataFrame, app_options: ao.AppOptions) -> pd.DataFrame:
        flb = FieldListBank(df)
        fields_to_rank = flb.overall_fields + flb.culture_fields + flb.quality_of_life_fields
        for field in fields_to_rank:
            df[f"{field}_rank"] = df[field].rank(ascending=False, method="min").astype(int)
        return df


# TODO move to config files or FieldListBank
culture_fit_codes = ["cf"]
quality_of_life_codes = ["ql"]
happy_planet_codes = ["hp"]
social_progress_codes = ["sp"]
human_freedom_codes = ["hf"]


def filter_by_codes(df, app_options, codes):
    for code in codes:
        threshold = getattr(app_options, f"{code}_score_min")
        df["satisfies_filters"] = df["satisfies_filters"] & (df[f"{code}_score"] > threshold)
    return df


class FiltersProcessor(DataProcessor):
    def process(self, df: pd.DataFrame, app_options: ao.AppOptions) -> pd.DataFrame:
        df["satisfies_filters"] = True
        if app_options.do_filter_culture_fit:
            df = filter_by_codes(df, app_options, culture_fit_codes)

        if app_options.do_filter_quality_of_life:
            df = filter_by_codes(df, app_options, quality_of_life_codes)

        if app_options.do_filter_happy_planet:
            df = filter_by_codes(df, app_options, happy_planet_codes)

        if app_options.do_filter_social_progress:
            df = filter_by_codes(df, app_options, social_progress_codes)

        if app_options.do_filter_freedom:
            df = filter_by_codes(df, app_options, human_freedom_codes)

        if app_options.do_filter_english:
            df["satisfies_filters"] = df["satisfies_filters"] & (df["english_ratio"] > app_options.english_ratio_min)

        if app_options.do_filter_temperature:
            df["satisfies_filters"] = (
                df["satisfies_filters"]
                & (df["average_temperature_celsius"] > app_options.average_temperature_celsius_min)
                & (df["average_temperature_celsius"] < app_options.average_temperature_celsius_max)
            )

        if app_options.do_filter_sunshine:
            df["satisfies_filters"] = (
                df["satisfies_filters"]
                & (df["average_sunshine_hours_per_day"] > app_options.average_sunshine_hours_per_day_min)
                & (df["average_sunshine_hours_per_day"] < app_options.average_sunshine_hours_per_day_max)
            )

        if app_options.do_filter_continents:
            df["satisfies_filters"] = df["satisfies_filters"] & (df["continent"].isin(app_options.continents))

        return df


@st.cache_data(ttl=app_config.TTL)
def process_data(app_options: ao.AppOptions | None = None):
    df = DATA.merged_df.copy()

    if app_options is None:
        app_options = ao.AppOptions()

    # TODO determine programmatically based on options
    # We only need the columns that participate in overall score and filters
    # TODO delegate dropna to each processor, ensuring that each one only drops rows as necessary for its operations
    flb = FieldListBank(df)
    required_columns = flb.quality_of_life_fields + flb.culture_fields + flb.climate_fields + flb.geography_fields
    df = df.dropna(subset=required_columns)

    seq = SequentialProcessor(
        [
            CultureFitProcessor(),
            LanguagePrevalenceProcessor(),
            OverallScoreProcessor(),
            RanksProcessor(),
        ]
    )
    df = seq.process(df, app_options)
    num_total = len(df)  # do this before filtering to get all rows
    fp = FiltersProcessor()
    df = fp.process(df, app_options)

    return df, num_total
