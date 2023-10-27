URL_BASE = "https://www.google.com/maps"


def get_google_maps_url(lat: float, lon: float) -> str:
    zoom_level = 5.0
    url = f"{URL_BASE}/@{lat},{lon},{zoom_level}z"
    return url
