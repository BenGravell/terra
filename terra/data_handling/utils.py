from terra.culture_fit import dimensions_info


# TODO move to a config file
# TODO harmonize with processing.FieldListBank
df_format_dict = {
    "country": "Country",
    "country_with_emoji": "Country with Emoji",
    "overall_score": "Overall Score",
    "cf_score": "Culture Fit Score",
    "ql_score": "Quality-of-Life Score",
    "hp_score": "Happy Planet Score",
    "sp_score": "Social Progress Score",
    "hf_score": "Human Freedom Score",
    "overall_score_rank": "Overall Score Rank",
    "cf_score_rank": "Culture Fit Score Rank",
    "ql_score_rank": "Quality-of-Life Score Rank",
    "hp_score_rank": "Happy Planet Score Rank",
    "sp_score_rank": "Social Progress Score Rank",
    "hf_score_rank": "Human Freedom Score Rank",
    "cf_score_weighted": "Culture Fit Score (weighted)",
    "ql_score_weighted": "Quality-of-Life Score (weighted)",
    "hp_score_weighted": "Happy Planet Score (weighted)",
    "sp_score_weighted": "Social Progress Score (weighted)",
    "hf_score_weighted": "Human Freedom Score (weighted)",
    "english_ratio": "English Speaking Ratio",
    "average_temperature_celsius": "Average Temperature (°C)",
    "average_sunshine_hours_per_day": "Average Daily Hours of Sunshine",
    "continent": "Continent",
    "satisfies_filters": "Satisfies Filters",
}
for dimension in dimensions_info.DIMENSIONS:
    df_format_dict[dimension] = dimensions_info.DIMENSIONS_INFO[dimension]["name"]
    df_format_dict[f"{dimension}_rank"] = f"{df_format_dict[dimension]} Rank"


def df_format_func(key):
    return df_format_dict[key]
