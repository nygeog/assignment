import json


def read_json(json_file):
    with open(json_file) as f:
        return json.load(f)


def pandas_keep_columns(
        df,
        column_list,
):
    return df[[x for x in df.columns if x in column_list]]
