module = True
type_err = True
other_err = True
try:
    from preprocess import clean_data, feature_data, multicol_data
    path = "../Data/nba2k-full.csv"
    df = multicol_data(feature_data, clean_data, path)
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
            return CheckResult.wrong('The function `multicol_data` was not found. Rename the function.')

        if not type_err:
            return CheckResult.wrong("Check the order of the input variables in the function and how they are called")

        if not other_err:
            return CheckResult.wrong("Probably problem with the execution of your functions. Refer to the examples.")

        if df is None:
            return CheckResult.wrong('The `multicol_data` function returns nothing but it should return a dataframe')

        if not isinstance(df, pd.DataFrame):
            return CheckResult.wrong(f'The `multicol_data` function returns a {type(df)} instead of a dataframe')

        if len(df.select_dtypes('number').drop(columns='salary').columns) < 3:
            return CheckResult.wrong('Incorrect number of features were dropped for multicollinearity')

        if len(df.select_dtypes('number').drop(columns='salary').columns) > 3:
            return CheckResult.wrong('The multicollinear feature is present in the dataframe')

        if sorted(df.select_dtypes('number').drop(columns='salary').columns.str.lower().tolist()) != sorted(['rating', 'experience', 'bmi']):
            return CheckResult.wrong('The incorrect feature was dropped for multicollinearity')

        return CheckResult.correct()


if __name__ == '__main__':
    Tests('Student').run_tests()
