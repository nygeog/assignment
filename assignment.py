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

        df.to_csv('data/processing/data.csv', index=False)

    # Part 1 Data Exploration and Evaluation
    #   clean columns, carry forward
    #       ‘loan_amnt’, ‘funded_amnt’, ‘term’, ‘int_rate’, ‘grade’,
    #       ‘annual_inc’, ‘issue_d’, ‘dti’, ‘revol_bal’, ‘total_pymnt’,
    #       ‘loan_status’
    #   perform any necessary cleaning and aggregations to explore and better
    #   understand the dataset.
    #
    #   Describe any assumptions you made to handle null variables and outliers.
    #
    #   Describe the distributions of the features.
    #
    #   Include two data visualizations and two summary statistics to support
    #   these findings.

    if 'select-historic-data' in steps:
        df = pd.read_csv(
            'data/processing/data.csv',
            low_memory=False,
            parse_dates=['issue_date']
        )

        df['max_date'] = pd.to_datetime(df.issue_date.max())
        df['months'] = (
                (df.max_date - df.issue_date) / np.timedelta64(1, 'M')
        ).astype(int)

        dfs = df[~(df.months < 36)]

    # Part 2 Business Analysis
    #   Assume a 36 month investment period for each loan, and
    #   exclude loans with less than 36 months of data available.
    # 1) What percentage of loans has been fully paid?
    # 2) When bucketed by year of origination and grade, which cohort has the
    #       highest rate of defaults? Here you may assume that any loan which
    #       was not fully paid had “defaulted”.
    # 3) When bucketed by year of origination and grade, what annualized rate of
    #       return have these loans generated on average?
    #
    #       For simplicity, use the following approximation:
    #       Annualized rate of return = (total_pymnt / funded_amnt) ^ (1/3) - 1

    # Part 3 Modeling
    # build a logistic regression model to predict loan defaults
    # Assume that
    #   (i) you are given the ability to invest in each loan independently
    #   (ii) you invest immediately following loan origination and hold to
    #       maturity (36 months)
    #   (iii) all loan fields that would be known upon origination are made
    #       available to you.

    # Was the model effective? Explain how you validated your model and describe
    # how you measure the performance of the model.
