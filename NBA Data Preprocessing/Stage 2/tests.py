module = True
type_err = True
other_err = True
try:
    from preprocess import clean_data, feature_data
    path = "../Data/nba2k-full.csv"
    df = feature_data(clean_data, path)
except ImportError:
    module = False
except TypeError:
    type_err = False
except Exception:
    other_err = False
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import pandas as pd


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if not module:
            return CheckResult.wrong('The function `feature_data` was not found. Rename the function.')

        if not type_err:
            return CheckResult.wrong("Check the order of the input variables in the function and how they are called")

        if not other_err:
            return CheckResult.wrong("Probably problem with the execution of your functions. Refer to the examples.")

        if df is None:
            return CheckResult.wrong('The `feature_data` function returns nothing but it should return a dataframe')

        if not isinstance(df, pd.DataFrame):
            return CheckResult.wrong(f'The `feature_data` function returns a {type(df)} instead of a dataframe')

        df_sample = df.sample(frac=1, random_state=43)

        if not any(df.columns.str.lower().str.contains('^age$')):
            return CheckResult.wrong('The age feature is absent')

        if not any(df.columns.str.lower().str.contains('^experience$')):
            return CheckResult.wrong('The experience feature is absent')

        if not any(df.columns.str.lower().str.contains('^bmi$')):
            return CheckResult.wrong('The bmi feature is absent')

        dropped_columns = ['version', 'b_day', 'draft_year', 'weight', 'height']

        for one_column in dropped_columns:
            if any(df.columns.str.lower().str.contains(one_column)):
                return CheckResult.wrong(f'{one_column} feature should be dropped')

        if list(df_sample.age.head()) != [27, 34, 24, 24, 31]:
            return CheckResult.wrong('The age feature calculation is incorrect')

        if list(df_sample.experience.head()) != [7, 14, 3, 4, 8]:
            return CheckResult.wrong('The experience feature calculation is incorrect')

        for res, ans in zip(list(df_sample.bmi.head()), [25.987736, 24.660336, 23.141399, 24.096678, 23.759027]):
            if not (ans - 0.2 < res < ans + 0.2):
                return CheckResult.wrong('The bmi feature calculation is incorrect')

        high_cardinality = ['full_name', 'draft_peak', 'college', 'jersey']

        count_card = 0
        for one_card in high_cardinality:
            if any(df.columns.str.lower().str.contains(one_card)):
                count_card += 1

        if count_card != 0:
            return CheckResult.wrong(f"Number of high cardinality feature(s) remaining in dataframe: {count_card}")

        return CheckResult.correct()


if __name__ == '__main__':
    Tests('Student').run_tests()
