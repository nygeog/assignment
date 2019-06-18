from tools.helpers import read_json, pandas_keep_columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]


def run_assignment(config):

    data = read_json(config)

    steps = data['steps']

    if "parse-data" in steps:
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

        len_pre_filter = len(df.index)

        df.to_csv('data/processing/data.csv', index=False)

        for col in ['annual_inc', 'revol_bal', 'dti']:
            col_mean = df[col].mean()
            col_std = df[col].std()

            df['z_score_{}'.format(col)] = df[col].apply(
                lambda x: (x - col_mean) / col_std
            )

            df = df[
                (
                    df[
                        'z_score_{}'.format(col)
                    ] < 3
                ) & (
                    df[
                        'z_score_{}'.format(col)
                    ] > -3
                )
            ].copy()

        df.to_csv('data/processing/data_filtered.csv', index=False)

        print(
            'pre-filter length: {}'.format(len_pre_filter),
            'post-filter length: {}'.format(len(df.index)),
            'dif. of: {}'.format(len_pre_filter - len(df.index)),
        )

    if "select-historic-data" in steps:
        df = pd.read_csv(
            'data/processing/data_filtered.csv',
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

        dfs['count'] = 1

        year_grade_default = dfs.groupby(
            ['year_grade', 'default'],
        ).agg({'count': 'sum'})

        year_grade = dfs.groupby(['year_grade']).agg({'count': 'sum'})

        grouped_year_defaults = year_grade_default.div(
            year_grade,
            level='year_grade',
        ) * 100

        grouped_year_defaults = grouped_year_defaults.reset_index()

        defaults = grouped_year_defaults[
            (grouped_year_defaults['default'] == 1)
        ]

        defaults.set_index("year_grade", drop=True, inplace=True)

        defaults_sort = defaults.sort_values(
            ['default', 'count'],
            ascending=[False, False],
        )

        defaults_sort.to_csv('data/processing/defaults_sort.csv', index=False)

        defaults_count = defaults_sort[['count']]

        defaults_count_top_20 = defaults_count.head(20)

        defaults_count_top_20.plot.bar(rot=0)

        plt.savefig('img/top_20_default_categories.png')

    if "annualized-rate-of-return" in steps:
        df = pd.read_csv(
            'data/processing/data_filtered.csv',
            low_memory=False,
            parse_dates=['issue_date']
        )

        dfs = df[~(df.months < 36)].copy()

        rate_return = dfs.groupby(
            ['year_grade'],
            as_index=False,
        ).agg({'annualized_rate_return': 'mean'})

        rate_return_sort = rate_return.sort_values(
            ['annualized_rate_return'],
            ascending=[False],
        )

        rate_return_sort.to_csv(
            'data/processing/rate_return_sort.csv',
            index=False,
        )

        rate_return_sort.set_index("year_grade", drop=True, inplace=True)

        rate_return_top = rate_return_sort.head(10)
        rate_return_bot = rate_return_sort.tail(10)

        rate_return_top.plot.bar(rot=0)
        plt.savefig('img/rate_return_top.png')

        rate_return_bot.plot.bar(rot=0)
        plt.savefig('img/rate_return_bot.png')
