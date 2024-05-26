import pandas as pd

from util import clean


def cleanse(data, save_to: str):
    # Drop rows with missing values
    data.dropna(inplace=True)

    # Define columns to use for cleaning
    features = ['w', 'h', 'k']

    # Clean data
    data = clean(data, features)

    # Save the cleaned data to a new CSV file
    data.to_csv(save_to, index=False)


if __name__ == '__main__':
    data_file = 'data.csv'
    save_file = 'pure.csv'

    # Load your CSV file into a DataFrame
    dataSet = pd.read_csv(data_file)

    # Clean & Save data
    cleanse(dataSet, save_file)
