import json
import os
from urllib.request import urlretrieve
import zipfile


def read_json(json_file):
    with open(json_file) as f:
        return json.load(f)


def pandas_keep_columns(
        df,
        column_list,
):
    return df[[x for x in df.columns if x in column_list]]


def create_directory(directory_folder):
    if not os.path.exists(directory_folder):
        os.makedirs(directory_folder)


def get_file(url, path, filename):
    urlretrieve(url, r'{}/{}.zip'.format(path, filename))
    return path, filename


def unzip_file(source_path, source_filename, dest_path, dest_folder_name):
    zip_ref = zipfile.ZipFile(
        r'{}/{}.zip'.format(
            source_path,
            source_filename,
        ),
        'r',
    )
    zip_ref.extractall(
        r'{}/{}'.format(
            dest_path,
            dest_folder_name,
        )
    )
    zip_ref.close()
    return r'{}/{}'.format(
        dest_path,
        dest_folder_name,
    )


def create_project_workspace():
    for i in [
        'data/input',
        'data/processing',
        'data/output',
    ]:
        create_directory('{}'.format(i))


def retrieve_data(data_url):
    print('    downloading data.')
    source_path, source_filename = get_file(
        data_url,
        'data/input',
        'kaggle_loan_data',

    )
    print('    unzipping data.')
    unzip_folder = unzip_file(
        source_path,
        source_filename,
        'data/input',
        'source_loan_data',
    )
    print('    unzipping data complete.')
    return unzip_folder
