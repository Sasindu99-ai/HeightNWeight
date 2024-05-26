import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

__all__ = [
    'clean', 'choose_model', 'choose_source', 'debug_state', 'split_data',
    'generate_file_name', 'save_model'
]


def clean(data, features: list):
    """
	Clean the dataset based on mean, median, Q1, and Q3
	"""
    for feature in features:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data[feature] >= lower_bound)
                    & (data[feature] <= upper_bound)]


def choose_model():
    """
	Choose a linear model to train
	:return: Type[LinearModel]
	"""
    choice = input(
        '\nChoose a linear model:\n\t1. LinearRegression Model\n\t2. LogisticRegression Model\n'
        '\nEnter your choice: ')
    if choice == '1':
        return LinearRegression
    if choice == '2':
        return LogisticRegression
    raise Exception('Invalid choice of model')


def choose_source() -> str:
    """
	Choose a source
	1. pure.csv
	2. group.csv
	"""
    choice = input(
        '\nChoose a source to train the model:\n\t1. pure.csv\n\t2. group.csv\n\nEnter your choice: '
    ).strip()
    if choice == '1':
        return 'pure.csv'
    if choice == '2':
        return 'group.csv'
    raise Exception('Invalid choice of source')


def debug_state() -> bool:
    """
	Turn on or off debug mode.
	When debug in on same data set is being used for training and testing.
	"""
    debug = input('Debug mode ON? (y/n): [y]').strip()
    if debug.lower() == 'y' or debug == '':
        return True
    if debug.lower() == 'n':
        return False
    raise Exception('Invalid debug mode')


def split_data(source: str, features: list, target: str, test_size: float,
               random_state: int | None) -> tuple:
    """
	Split the dataset into training and testing.
	:return: tuple of training and testing.
	"""
    # Load cleaned data from chosen source
    data = pd.read_csv(source)

    # Split data into features and target
    x = data[features]
    y = data[target]

    # Split data into train and test sets
    return train_test_split(x,
                            y,
                            test_size=test_size,
                            random_state=random_state)


def generate_file_name(model, source: str, debug: bool) -> str:
    return f"{model.__name__}-{source.split('.')[0]}{'-debug' if debug else ''}.pickle"


def save_model(model, file_name: str):
    with open(f'trained_models/{file_name}', 'wb') as f:
        pickle.dump(model, f)
