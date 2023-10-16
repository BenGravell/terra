"""Wrangle city sunshine hours data to country average aggregate data.

NOTE: ad-hoc adjustments to the output file country_sunshine_hours_per_day.csv were made after running this script to be
compatible with the other data in this project. Therefore, do not run this script without making the same adjustments.
"""

import calendar

import pandas as pd


df = pd.read_csv("./data/city_sunshine_hours.csv")

gdf = df.groupby("country").median(numeric_only=True)

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

days_in_months = {}
for i in range(12):
    month = months[i]
    month_idx = i + 1
    num_days = calendar.monthrange(2023, month_idx)[1]
    days_in_months[month] = num_days

mdf = pd.DataFrame.from_dict(days_in_months, orient="index")
mdf.columns = ["num_days"]
mdf = mdf.transpose()

adf = gdf.copy()
for month in months:
    adf[month] = adf[month] / mdf[month].item()

adf["year"] = adf["year"] / sum(val for val in days_in_months.values())

adf.round(2).to_csv("data/country_sunshine_hours_per_day_NEEDS_EDITS.csv")
