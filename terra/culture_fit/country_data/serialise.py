from terra.culture_fit.country_data import types


def json_to_countries(raw_data: types.JSONType) -> types.Countries:
    return [serialise_country(country) for country in raw_data]


def serialise_country(raw_data: types.JSONType) -> types.CountryInfo:
    return types.CountryInfo(
        raw_data["id"],
        raw_data["title"].replace("*", ""),
        raw_data["slug"],
        int(raw_data["pdi"]) if raw_data["pdi"] else None,
        int(raw_data["idv"]) if raw_data["idv"] else None,
        int(raw_data["mas"]) if raw_data["mas"] else None,
        int(raw_data["uai"]) if raw_data["uai"] else None,
        int(raw_data["lto"]) if raw_data["lto"] else None,
        int(raw_data["ind"]) if raw_data["ind"] else None,
        raw_data["adjective"],
    )
