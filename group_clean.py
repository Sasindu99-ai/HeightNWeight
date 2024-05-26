import pandas as pd

from util import clean


def cleanse(data, save_to: str):
    # Define columns to use for cleaning
    features = ['w', 'h', 'k']

    # Separate the data into male and female groups
    males = data[data['gender'] == 1]
    females = data[data['gender'] == -1]

    # Clean each group separately
    cleaned_males = clean(males, features)
    cleaned_females = clean(females, features)

    # Concatenate the cleaned groups back together
    cleaned_data = pd.concat([cleaned_males, cleaned_females])

    cleaned_data.to_csv(save_to, index=False)


if __name__ == '__main__':
    data_file = 'data.csv'
    save_file = 'group.csv'

    # Load your CSV file into a DataFrame
    dataSet = pd.read_csv(data_file)

    # Clean & Save data
    cleanse(dataSet, save_file)
