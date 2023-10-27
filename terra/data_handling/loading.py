import dataclasses
from abc import ABC, abstractmethod
from typing import Any
import json

import pandas as pd

from terra.culture_fit import country_data, dimensions_info
from terra.resource_utils import get_data_file_path


def read_data_csv_to_pandas(filename: str) -> pd.DataFrame:
    path = get_data_file_path(filename)
    return pd.read_csv(path)


def read_data_json(filename: str) -> Any:
    path = get_data_file_path(filename)

    with open(path, "r") as f:
        return json.load(f)


class Dataset(ABC):
    def __init__(self) -> None:
        self.name: str | None = None
        self.source_filename: str | None = None

    @abstractmethod
    def load_raw(self):
        pass

    def preprocess(self):
        pass

    def load(self):
        self.load_raw()
        self.preprocess()


class CultureFitDataset(Dataset):
    def __init__(self) -> None:
        self.name = "Culture Fit"
        self.source_filename = None  # TODO add real filename

    def load_raw(self):
        self.d = country_data.get_country_dict()  # culture_fit_data_dict

    def preprocess(self):
        # Remove countries that do not have all dimensions populated
        country_names_to_remove = set()
        for country_name, country_info in self.d.items():
            for dimension in dimensions_info.DIMENSIONS:
                val = getattr(country_info, dimension)
                if val is None or val < 0:
                    country_names_to_remove.add(country_name)

        for country_name in country_names_to_remove:
            self.d.pop(country_name)

        self.df = pd.DataFrame.from_dict(self.d, orient="index")[dimensions_info.DIMENSIONS]
        # Undo 100X scaling
        self.df *= 0.01
        # Move country to column
        self.df = self.df.rename_axis("country").reset_index()


class DataFrameDataset(Dataset):
    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None

    def load_raw(self):
        self.df = read_data_csv_to_pandas(self.source_filename)

    def max_normalize(self, key: str):
        # Normalize by max value achieved in the dataset
        self.df[key] /= self.df[key].max()


class HappyPlanetDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Happy Planet"
        self.source_filename = "happy_planet_index_2019.csv"

    def preprocess(self):
        self.df = self.df[["country", "HPI"]]
        self.df = self.df.rename(columns={"HPI": "hp_score"})
        # Undo 100X scaling
        self.df["hp_score"] *= 0.01
        self.max_normalize("hp_score")


class SocialProgressDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Social Progress"
        self.source_filename = "social_progress_index_2022.csv"

    def preprocess(self):
        # Pick out just the columns we need and rename country column
        social_progress_cols_keep = ["country", "Social Progress Score"]
        self.df = self.df[social_progress_cols_keep]
        self.df = self.df.set_index("country")
        # Undo 100X scaling
        self.df *= 0.01
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={"Social Progress Score": "sp_score"})
        self.max_normalize("sp_score")


class HumanFreedomDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Human Freedom"
        self.source_filename = "human-freedom-index-2022.csv"

    def preprocess(self):
        self.max_normalize("hf_score")
        year_min, year_max = 2015, 2020
        human_freedom_df_year_filtered = self.df.query(f"{year_min} <= year <= {year_max}")
        freedom_score_cols = ["hf_score"]
        self.df = human_freedom_df_year_filtered.groupby(["country"])[freedom_score_cols].mean()
        self.df = self.df.reset_index()


class EnglishDataset(DataFrameDataset):
    def __init__(self):
        self.name = "English"
        self.source_filename = "english_speaking.csv"

    def preprocess(self):
        self.df = self.df[["country", "english_ratio"]]


class TemperatureDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Temperature"
        self.source_filename = "country_temperature.csv"


class SunshineDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Sunshine"
        self.source_filename = "country_sunshine_hours_per_day.csv"

    def preprocess(self):
        self.df = self.df.rename(columns={"year": "average_sunshine_hours_per_day"})
        self.df = self.df[["country", "average_sunshine_hours_per_day"]]


class CoordinatesDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Coordinates"
        self.source_filename = "country_coords.csv"


class ContinentsDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Continents"
        self.source_filename = "country_continents.csv"


class CodesAlpha3Dataset(DataFrameDataset):
    def __init__(self):
        self.name = "Codes Alpha 3"
        self.source_filename = "country_codes_alpha_3.csv"


class FlagEmojiDataset(DataFrameDataset):
    def __init__(self):
        self.name = "Flag Emoji"
        self.source_filename = "country_flag_emoji.csv"


# TODO move to utils
def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str] | str,
    *args,
    **kwargs,
) -> pd.DataFrame | None:
    """Merge left and right DataFrames on the first column from columns that is present in both DataFrames."""

    # Handle single column case
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column in left.columns and column in right.columns:
            return left.merge(right, on=column, *args, **kwargs)
    return None


def safe_merge_multi(
    dfs: list[pd.DataFrame],
    columns: list[str] | str,
    *args,
    **kwargs,
) -> pd.DataFrame | None:
    merged_df = dfs[0].copy()
    for df in dfs[1:]:
        merged_df = safe_merge(merged_df, df, columns, *args, **kwargs)
        if merged_df is None:
            return None
    return merged_df


@dataclasses.dataclass
class DatasetBundle:
    datasets: list[Dataset]

    def load(self):
        for dataset in self.datasets:
            dataset.load()

    def asdict(self) -> dict[str, Dataset]:
        return {dataset.name: dataset for dataset in self.datasets}

    @property
    def dfs(self) -> list[pd.DataFrame | None]:
        return [getattr(dataset, "df") for dataset in self.datasets]


# TODO use maintenance script to create merged data just once & load that in a simple DataFrameDataset in the app
@dataclasses.dataclass
class MergedDatasetBundle(DatasetBundle):
    def __post_init__(self) -> None:
        self.merged_df: pd.DataFrame | None = None

        self.load()
        self.merge()
        self.postprocess()

    def merge(self):
        self.merged_df = safe_merge_multi(self.dfs, ["country", "country_code_alpha_3"], how="outer")

        # Remove rows that do not have a known country
        self.merged_df = self.merged_df.dropna(subset=["country"])

    def postprocess(self):
        # Add column with country and emoji
        df = self.merged_df
        mask = df["emoji"].notna()
        df["country_with_emoji"] = df["country"]
        df.loc[mask, "country_with_emoji"] = df["country"] + " (" + df["emoji"] + ")"


DATA = MergedDatasetBundle(
    [
        CultureFitDataset(),
        HappyPlanetDataset(),
        SocialProgressDataset(),
        HumanFreedomDataset(),
        EnglishDataset(),
        TemperatureDataset(),
        SunshineDataset(),
        CoordinatesDataset(),
        ContinentsDataset(),
        CodesAlpha3Dataset(),
        FlagEmojiDataset(),
    ]
)
DATA_DICT = DATA.asdict()

# TODO move to unit test
assert DATA.merged_df["country"].nunique() == len(DATA.merged_df["country"])


COUNTRY_FLAG_IMAGE_URLS = read_data_json("country_flag_image_urls.json")


if __name__ == "__main__":
    print(DATA)
