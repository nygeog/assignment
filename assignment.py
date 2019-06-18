from tools.helpers import read_json, pandas_keep_columns, retrieve_data
from tools.helpers import create_project_workspace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]


def run_assignment(config):

    data = read_json(config)

    steps = data['steps']

    if "setup-workspace-retrieve-and-parse-data" in steps:
        create_project_workspace()

        unzip_folder = retrieve_data(data["download_link"])

        df = pandas_keep_columns(
            pd.read_csv(
                '{}/{}'.format(unzip_folder, data['input_data_file_name']),
                low_memory=False,
            ),
            data['keep_columns'],
        )

        print('    creating value added fields for data.')

        df['issue_date'] = pd.to_datetime(df['issue_d'])
        df['issue_year'] = df['issue_date'].dt.year
        df['issue_month'] = df['issue_date'].dt.month
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

        print('    filtering fields for data.')

        for col in ['annual_inc', 'revol_bal', 'dti']:
            col_mean = df[col].mean()
            col_std = df[col].std()

            df['z_score_{}'.format(col)] = df[col].apply(
                lambda v: (v - col_mean) / col_std
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
            ].copy()  # filter data for spec. cols, z-score'd

        df.to_csv('data/processing/data_filtered.csv', index=False)

        print(
            '    pre-filter length: {}'.format(len_pre_filter),
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
            '    {:.2f}% of loans Fully Paid excluding < 36 months'.format(
                pct_total_fully_paid * 100
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

        defaults_sort.to_csv('data/processing/defaults_sort.csv')

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

    if 'logistic-regression' in steps:
        df = pd.read_csv(
            'data/processing/data_filtered.csv',
            low_memory=False,
        )

        df = df.drop(
            [
                'issue_date',  # parsed out to month and year
                'z_score_annual_inc',  # removing these, for filtering part 1
                'z_score_revol_bal',  # removing these, for filtering part 1
                'z_score_dti',  # removing these, for filtering part 1
                'year_grade',  # year and grade are also own columns
                'loan_status',  # this is dummy in 'default' col
                'max_date',  # removing, for filtering in part 2
                'issue_d',  # original date field, removed sim. to 'issue_date'
            ],
            axis=1,
        )

        one_hot = pd.get_dummies(df)

        x = one_hot.drop('default', axis=1)
        y = df[['default']]

        # x.to_csv('data/processing/data_filtered_train.csv')

        x_train, x_test, y_train, y_test = train_test_split(
            x, y.values.ravel(),
            test_size=0.3,
            random_state=0,
        )
        model = LogisticRegression(solver='lbfgs', max_iter=500)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        print(
            '    Accuracy of logistic regression on test: {:.2f}%'.format(
                model.score(x_test, y_test) * 100
            )
        )

        confusion_matrix_plot = confusion_matrix(y_test, y_pred)
        print('    Confusion Matrix Plot')
        print(confusion_matrix_plot)

        print('    Classification Report')
        print(classification_report(y_test, y_pred))

        logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
        fpr, tpr, thresholds = roc_curve(
            y_test,
            model.predict_proba(x_test)[:, 1],
        )
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            label='Logistic Regression (area = {:.2f})'.format(logit_roc_auc),
        )
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('img/log_roc_split.png')
        # plt.show()
