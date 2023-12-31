import pandas as pd
from scipy.spatial import distance
from sklearn import decomposition

from terra.culture_fit.country_data import types

HOFSTEDE_DIMENSIONS = ["pdi", "idv", "mas", "uai", "lto", "ind"]
AVAILABLE_DISTANCES = {
    "Euclidean": distance.euclidean,
    "Cosine": distance.cosine,
    "Manhattan": distance.cityblock,
    "Correlation": distance.correlation,
}

AVAILABLE_DECOMPOSITION = {
    "PCA": decomposition.PCA,
    "FastICA": decomposition.FastICA,
    "NMF": decomposition.NMF,
    "MiniBatchSparsePCA": decomposition.MiniBatchSparsePCA,
    "SparsePCA": decomposition.SparsePCA,
    "TruncatedSVD": decomposition.TruncatedSVD,
}

TO_PERCENT = 100.0
SQUARE = 2


def compute_dimensions(countries: types.Countries) -> pd.DataFrame:
    index = [country.title for country in countries]
    dimensions = {}
    for dimension in HOFSTEDE_DIMENSIONS:
        row = []
        for country in countries:
            row.append(max(getattr(country, dimension) or 0, 0))
        dimensions[dimension] = row
    return pd.DataFrame(dimensions, index=index).transpose()


def compute_distance(country_from: types.CountryInfo, country_to: types.CountryInfo, distance_metric: str) -> float:
    from_array = [max(getattr(country_from, dimension) or 0, 0) for dimension in HOFSTEDE_DIMENSIONS]
    to_array = [max(getattr(country_to, dimension) or 0, 0) for dimension in HOFSTEDE_DIMENSIONS]
    return AVAILABLE_DISTANCES[distance_metric](from_array, to_array)


def compute_distances(
    countries_from: types.Countries, countries_to: types.Countries, distance_metric: str
) -> tuple[pd.DataFrame, float]:
    index = [country.title for country in countries_to]
    distances = {}
    max_distance = 0
    for country_from in countries_from:
        row = []
        for country_to in countries_to:
            distance = compute_distance(country_from, country_to, distance_metric)
            max_distance = max(max_distance, distance)
            row.append(distance)
        distances[country_from.title] = row
    return pd.DataFrame(distances, index=index), max_distance


def normalise_distance_matrix(distances: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    return distances.applymap(lambda x: x / max_distance)


def generate_2d_coords(dimensions: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    algo = AVAILABLE_DECOMPOSITION[algorithm]
    reduced = algo(n_components=2)
    ret = pd.DataFrame(reduced.fit_transform(dimensions.transpose()), index=dimensions.columns)
    return ret
