from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import pandas as pd
import os

module = True
type_err = True
other_err = True
try:
    from preprocess import clean_data, feature_data, multicol_data
    path = "../Data/nba2k-full.csv"
    df = multicol_data(feature_data(clean_data(path)))
except ImportError:
    module = False
    clean_data = None
    feature_data = None
    multicol_data = None
except TypeError as type_err_exc:
    type_err_exc_message = type_err_exc
    type_err = False
except Exception as other_exc:
    other_exc_message = other_exc
    other_err = False


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if not os.path.exists('preprocess.py'):
            return CheckResult.wrong('The file `preprocess.py` is not found. Your solution should be located there.\n'
                                  'Please do not rename the file.')

        if not module:
            return CheckResult.wrong('The function `multicol_data` was not found in your solution')

        if not type_err:
            return CheckResult.wrong(f"An error occurred during execution of your solution.\n"
                                     f"The function `multicol_data` should take one input parameter: DataFrame returned by `feature_data` function.\n"
                                     f"An internal error message:\n{type_err_exc_message}")

        if not other_err:
            return CheckResult.wrong(f"An error occurred during execution of `multicol_data` function.\n"
                                     f"The error message:\n{other_exc_message}\n\n"
                                     f"Refer to the Objectives and Examples sections.")

        if df is None:
            return CheckResult.wrong('The `multicol_data` function returns nothing while it should return a DataFrame')

        if not isinstance(df, pd.DataFrame):
            return CheckResult.wrong(f'The `multicol_data` function returns a {type(df)} instead of pandas DataFrame')

        if "salary" not in df:
            return CheckResult.wrong("The salary variable is absent")

        if len(df.select_dtypes('number').drop(columns='salary').columns) < 3:
            return CheckResult.wrong('Incorrect number of features were dropped for multicollinearity')

        if len(df.select_dtypes('number').drop(columns='salary').columns) > 3:
            return CheckResult.wrong('Multicollinearity is still present in the DataFrame')

        if sorted(df.select_dtypes('number').drop(columns='salary').columns.str.lower().tolist()) != sorted(['rating', 'experience', 'bmi']):
            return CheckResult.wrong(f"Your set of numerical features is currently as follows: {df.select_dtypes('number').drop(columns='salary').columns.tolist()} plus 'salary'.\n"
                                     f"This set is wrong, probably an incorrect feature was dropped for multicollinearity.")

        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
