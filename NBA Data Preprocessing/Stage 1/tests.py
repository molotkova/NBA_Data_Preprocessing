from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import pandas as pd

module = True
type_err = True
other_err = True
try:
    from preprocess import clean_data
    path = "../Data/nba2k-full.csv"
    df = clean_data(path)
except ImportError:
    clean_data = None
    module = False
except TypeError:
    type_err = False
except Exception:
    other_err = False


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if not module:
            return CheckResult.wrong('The function `clean_data` was not found in your solution.')

        if not type_err:
            return CheckResult.wrong("Do not forget the input parameter: path")

        if not other_err:
            return CheckResult.wrong("An error occurred during execution of `clean_data` function. Refer to the Objectives and Examples sections.")

        if df is None:
            return CheckResult.wrong('The `clean_data` function returns nothing while it should return a DataFrame')

        if not isinstance(df, pd.DataFrame):
            return CheckResult.wrong(f'The `clean_data` function returns a {type(df)} instead of pandas DataFrame')

        col_used = ['b_day', 'draft_year', 'team', 'height', 'weight', 'salary', 'country', 'draft_round']

        for one_col in col_used:
            if one_col not in df:
                return CheckResult.wrong(f'The column {one_col} is not in the DataFrame')

        df_datetime = df.select_dtypes(include=['datetimetz', 'datetime']).columns.tolist()
        if sorted(df_datetime) != sorted(['b_day', 'draft_year']):
            return CheckResult.wrong('Convert `b_day` and `draft_year` columns to datetime objects')

        if df.team.isna().sum() != 0:
            return CheckResult.wrong('There are missing values in the `team` column')

        if df.team.str.contains("No Team").sum() == 0:
            return CheckResult.wrong("Replace missing values with `No Team` in the team column")

        df_floats = df.select_dtypes(include=['float']).columns
        if sorted(df_floats) != sorted(['height', 'weight', 'salary']):
            return CheckResult.wrong('The height, weight, and salary columns should be float')

        if list(df.loc[0, ['height', 'weight']]) != [2.06, 113.4]:
            return CheckResult.wrong('The height should be in meters and the weight in kg')

        df_country = list(df.country.unique())
        if sorted(df_country) != sorted(['USA', 'Not-USA']):
            return CheckResult.wrong('The country columns should have two unique categories: USA and Not-USA')

        if df.draft_round.str.contains('0').sum() == 0:
            return CheckResult.wrong('Change `Undrafted` to `"0"` in the draft_round column')

        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
