import os
import pickle

import numpy as np

from pre import generate_k


# Function to load the model from a pickle file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Function to predict the outcome using the model
def predict(model, weight, height, k):
    # Create an array for the input features
    input_features = np.array([[weight, height, k]])
    # Predict the outcome
    outcome = model.predict(input_features)
    return outcome


# Function to print the model's coefficients and intercept
def print_model_details(model):
    if hasattr(model, 'coef_'):
        print('Model Coefficients:', model.coef_)
    if hasattr(model, 'intercept_'):
        print('Model Intercept:', model.intercept_)


def main(folder: str):
    # List available pickle files in the current directory
    pickle_files = [
        file for file in os.listdir(folder) if file.endswith('.pickle')
    ]
    if not pickle_files:
        print('No pickle files found in the current directory.')
        return

    # Display the available pickle files
    print('Available models:')
    for idx, file in enumerate(pickle_files):
        print(f'{idx + 1}. {file}')

    # Choose a pickle file
    choice = int(input('Enter the number of the model you want to load: ')) - 1
    if choice < 0 or choice >= len(pickle_files):
        print('Invalid choice.')
        return

    # Load the chosen model
    model_file = pickle_files[choice]
    model = load_model(f'{folder}/{model_file}')
    print(f'Loaded model: {model_file}')

    # Enter weight and height
    weight = float(input('Enter weight (kg): '))
    height = float(input('Enter height (cm): '))

    k = generate_k(weight, height)

    # Predict the outcome
    outcome = predict(model, weight, height, k)
    print(f'Predicted outcome: {outcome}')

    if outcome[0] == 1:
        print('Your gender is predicted as Male')
    elif outcome[0] == -1:
        print('Your gender is predicted as Female')
    else:
        print('Model could not identify your gender correctly')
        if outcome[0] > 0:
            print('However, your gender can be predicted as Male')
        elif outcome[0] < 0:
            print('However, your gender can be predicted as Female')

    # Print the model's coefficients and intercept
    print_model_details(model)


if __name__ == '__main__':
    models = 'trained_models_backup'
    main(models)
