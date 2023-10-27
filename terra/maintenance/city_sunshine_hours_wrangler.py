"""Wrangle city sunshine hours data to country average aggregate data."""

import calendar

import pandas as pd

from terra.resource_utils import get_data_file_path
from terra.data_handling.loading import read_data_csv_to_pandas


# Read
df = read_data_csv_to_pandas("city_sunshine_hours.csv")

# Process
grouped_df = df.groupby("country").median(numeric_only=True)

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

days_in_months = {}
for i in range(12):
    month = months[i]
    month_idx = i + 1
    num_days = calendar.monthrange(2023, month_idx)[1]
    days_in_months[month] = num_days

month_df = pd.DataFrame.from_dict(days_in_months, orient="index")
month_df.columns = ["num_days"]
month_df = month_df.transpose()

year_df = grouped_df.copy()
for month in months:
    year_df[month] = year_df[month] / month_df[month].item()

year_df["year"] = year_df["year"] / sum(val for val in days_in_months.values())

# Write
year_df.round(2).to_csv(get_data_file_path("country_sunshine_hours_per_day.csv"))
