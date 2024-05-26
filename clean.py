import pandas as pd


def clean(data, features: list):
    """Clean the dataset based on mean, median, Q1, and Q3"""
    for feature in features:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data[feature] >= lower_bound)
                    & (data[feature] <= upper_bound)]


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
