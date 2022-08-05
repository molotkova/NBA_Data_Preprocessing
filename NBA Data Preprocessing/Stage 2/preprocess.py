import pandas as pd
import numpy as np
import os
import requests


def clean_data(train_path):

    # Import data and parse b_day and draft_year as datetime
    df = pd.read_csv(train_path, parse_dates=['b_day', 'draft_year'])
    # Change NaN values in team colum to `No Team`
    df.loc[df.team.isna(), 'team'] = "No Team"
    # Take `height` in meters and convert to float
    df['height'] = df.height.str.split('/').str[-1].str.strip().astype(float)
    # Take weight in kg and convert to float
    df['weight'] = df.weight.str.split('/').str[-1].str.strip().str.split('kg').str[0].astype(float)
    # Remove `$` from salary and convert to float
    df['salary'] = df.salary.str.replace("$", "", regex=False).astype(float)
    # Change country to USA and Not-USA
    df.loc[df.country != 'USA', 'country'] = "Not-USA"
    # Change undrafted in draft round column to `"0"`
    df.loc[df.draft_round == 'Undrafted', 'draft_round'] = "0"

    return df


def feature_data(df):

    # There are two versions. Get them and parse as datetime
    df['version'] = df.version.str.replace('NBA2k', '20').astype(np.datetime64)
    # Get `age` feature
    df['age'] = df.version.dt.year - df.b_day.dt.year
    # Get `experience` feature
    df['experience'] = df.version.dt.year - df.draft_year.dt.year
    # Drop columns that are no longer needed: `version`, `b_day`, `draft_year`
    df.drop(columns=['version', 'b_day', 'draft_year'], inplace=True)

    # Get `bmi`
    df['bmi'] = df.weight / df.height**2

    # Drop columns that are no longer needed: `weight`, `height`
    df.drop(columns=['weight', 'height'], inplace=True)

    # Remove high cadinality columns: `full_name`, `draft_peak` and `college`
    df.drop(columns=['full_name', 'draft_peak', 'college', 'jersey'], inplace=True)

    return df

#
# def feature_data(func, path):
#
#     df = func(path)
#
#     # There are two versions. Get them and parse as datetime
#     df['version'] = df.version.str.replace('NBA2k', '20').astype(np.datetime64)
#     # Get `age` feature
#     df['age'] = df.version.dt.year - df.b_day.dt.year
#     # Get `experience` feature
#     df['experience'] = df.version.dt.year - df.draft_year.dt.year
#     # Drop columns that are no longer needed: `version`, `b_day`, `draft_year`
#     df.drop(columns=['version', 'b_day', 'draft_year'], inplace=True)
#
#     # Get `bmi`
#     df['bmi'] = df.weight / df.height**2
#
#     # Drop columns that are no longer needed: `weight`, `height`
#     df.drop(columns=['weight', 'height'], inplace=True)
#
#     # Remove high cadinality columns: `full_name`, `draft_peak` and `college`
#     df.drop(columns=['full_name', 'draft_peak', 'college', 'jersey'], inplace=True)
#
#     return df


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'nba2k-full.csv' not in os.listdir('../Data'):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/nba2k-full.csv', 'wb').write(r.content)
        print('Loaded.')

    # path = "../Data/nba2k-full.csv"
    # df = feature_data(clean_data, path)
    # df_ = df.sample(frac=1, random_state=43)
    # print(any(df_.columns.str.lower().str.contains('ages')))
    # print(df_[['age', 'experience', 'bmi']].head())

