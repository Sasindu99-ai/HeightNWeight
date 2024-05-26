from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from pre import TEST_SIZE, features, state, target
from util import choose_model, choose_source, debug_state, split_data


def main(linear_model, source: str, test_size: float,
         random_state: int | None) -> tuple:
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = split_data(source, features, target,
                                                  test_size, random_state)

    # Feature scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the linear model
    model = linear_model()
    model.fit(x_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(x_test_scaled)

    # Calculate accuracy
    return r2_score(y_test, y_pred), model


if __name__ == '__main__':
    print('Linear Model\n')

    # Choose whether to run on debug mode or not
    DEBUG = debug_state()

    accuracy, linear = main(choose_model(), choose_source(), TEST_SIZE,
                            state(DEBUG))

    print('\nAccuracy of the trained model:', accuracy)
