# HeightNWeight

---

#### Description

Predict gender by taking weight and height as inputs.

#### Dataset

The dataset consists of the following columns:
- `w` (weight in kg)
- `h` (height in cm)
- `k` (generated feature calculated as `k = w * 0.02 + h * 0.06 - 10.29`)
- `gender` (1 means Male, -1 means Female)

## 1. Setup

Ensure you have `conda` and `poetry` installed.

If you don't have a conda environment set up with Python version 3.12.3, you can create one using the following command, where `ENV_NAME` is `tensor`:

```sh
conda create -n tensor python==3.12.3
```

Activate the environment:

```sh
conda activate tensor
```

Use `poetry` to install dependencies:

```sh
poetry install
```

## 2. Data Cleaning

### First Method: Clean the whole dataset

To clean the dataset and save it as `pure.csv`, run the following command:

```sh
python clean.py
```

This script loads the data from `data.csv`, performs data cleaning operations by removing outliers based on the interquartile range (IQR) for the entire dataset, and saves the cleaned data to `pure.csv`.

### Second Method: Separate Cleaning for Males and Females

To clean the dataset separately for males and females and save the combined cleaned data as `group.csv`, run the following command:

```sh
python group_clean.py
```

This script loads the data from `data.csv`, cleans the data for males and females separately, and saves the combined cleaned data to `group.csv`.

#### Reason for the New Method

The new method was tried to account for potential differences in the distribution of features between males and females. By cleaning the data separately for each gender, we aim to preserve the integrity and variability of each group, which may improve the performance of the model trained on this data.

## License

This project is licensed under the [MIT License](LICENSE).
