from tools.helpers import read_json, pandas_keep_columns
import pandas as pd
import numpy as np


def run_assignment(config):

    data = read_json(config)

    steps = data['steps']

    if 'parse-data' in steps:
        df = pandas_keep_columns(
            pd.read_csv(
                data['input_data'],
                low_memory=False,
            ),
            data['keep_columns'],
        )

        df['issue_date'] = pd.to_datetime(df['issue_d'])
        df['issue_year'] = df['issue_date'].dt.year
        df['max_date'] = pd.to_datetime(df.issue_date.max())
        df['months'] = (
                (df.max_date - df.issue_date) / np.timedelta64(1, 'M')
        ).astype(int)

        df['year_grade'] = df.issue_year.apply(str) + '-' + df.grade
        df['default'] = np.where(
            df['loan_status'] == 'Fully Paid',
            0,
            1,
        )

        df['annualized_rate_return'] = (
            np.power(df['total_pymnt'] / df['funded_amnt'], 1/3) - 1.0
        )

        # MAYBE FILTER OUT ANNUAL INCOME - OUTLIERS?

        df.to_csv('data/processing/data.csv', index=False)



    if 'select-historic-data' in steps:
        df = pd.read_csv(
            'data/processing/data.csv',
            low_memory=False,
            parse_dates=['issue_date']
        )

        dfs = df[~(df.months < 36)].copy()

        fully_paid_count = dfs.groupby(
            ['loan_status']
        ).term.count()['Fully Paid']

        pct_total_fully_paid = fully_paid_count / (len(dfs.index))

        print(
            '{} pct. of loans Fully Paid excluding < 36 months'.format(
                pct_total_fully_paid
            )
        )



