from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import pandas as pd

module = True
type_err = True
other_err = True
try:
    from preprocess import clean_data, feature_data, multicol_data, transform_data
    path = "../Data/nba2k-full.csv"
    answer = transform_data(multicol_data, feature_data, clean_data, path)
except ImportError:
    module = False
    clean_data = None
    feature_data = None
    multicol_data = None
    transform_data = None
except TypeError:
    type_err = False
except Exception:
    other_err = False


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if not module:
            return CheckResult.wrong('The function `transform_data` was not found in your solution')

        if not type_err:
            return CheckResult.wrong("Check the order of the input variables in the function and how they are called")

        if not other_err:
            return CheckResult.wrong("An error occurred during execution of `transform_data` function. Refer to the Objectives and Examples sections.")

        if answer is None:
            return CheckResult.wrong('The `transform_data` function returns nothing while it should return X DataFrame and y series')

        if len(answer) != 2:
            return CheckResult.wrong("The transform_data function should return X and y")

        first, second = answer

        if isinstance(first, pd.DataFrame) and isinstance(second, pd.Series):
            X, y = answer
        elif isinstance(first, pd.Series) and isinstance(second, pd.DataFrame):
            return CheckResult.wrong('Return X DataFrame and y series as X,y not y, X')
        else:
            return CheckResult.wrong('Return X as a DataFrame and y as a series')

        if X.shape != (439, 46):
            return CheckResult.wrong('X DataFrame has wrong shape')

        if y.shape != (439,):
            return CheckResult.wrong('y series has wrong shape')

        if list(X.columns.str.lower())[:3] != ['rating', 'experience', 'bmi']:
            return CheckResult('The numerical columns are arranged in the following order: rating, experience, bmi')

        if list(X.columns[3:]) != ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
                                   'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
                                   'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
                                   'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
                                   'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
                                   'No Team', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
                                   'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
                                   'Utah Jazz', 'Washington Wizards', 'C', 'C-F', 'F', 'F-C', 'F-G', 'G', 'G-F',
                                   'Not-USA', 'USA', '0', '1', '2']:
            return CheckResult.wrong("The categorical columns are in the following order before one-hot encoding: team, position, country, draft round")

        if list(X.columns) != ['rating', 'experience', 'bmi', 'Atlanta Hawks', 'Boston Celtics',
                               'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers',
                               'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors',
                               'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers', 'Los Angeles Lakers',
                               'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves',
                               'New Orleans Pelicans', 'New York Knicks', 'No Team', 'Oklahoma City Thunder',
                               'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers',
                               'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz',
                               'Washington Wizards', 'C', 'C-F', 'F', 'F-C', 'F-G', 'G', 'G-F', 'Not-USA', 'USA',
                               '0', '1', '2']:
            return CheckResult.wrong("Put numerical features before categorical features during concatenation")

        scaled_ans = [3.2352194717791973, 2.7598866876192636, 1.3201454530024874]
        student_ans = X.head(1).values.tolist()[0][:3]

        for one_scale, one_student in zip(scaled_ans, student_ans):
            if not (one_scale - .2 < one_student < one_scale + 0.2):
                return CheckResult.wrong('Standard Scaler transformation is done incorrectly')

        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
