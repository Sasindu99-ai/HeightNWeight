from sklearn.linear_model import LinearRegression, LogisticRegression

from linear import main
from pre import TEST_ROUNDS, TEST_SIZE, state
from util import generate_file_name, save_model


def save_best_model(model, source: str, rounds, debug: bool):
    """
	Run training rounds and saves the model which has the best accuracy
	:param model: model to be trained
	:param source: data source
	:param rounds: number of rounds
	:param debug: debug mode
	:return: None
	"""
    model_file_name = generate_file_name(model, source, debug)
    best_accuracy = 0

    for _ in range(rounds):
        accuracy, linear = main(model, source, TEST_SIZE, state(debug))
        if best_accuracy < accuracy:

            # Change the best accuracy to new value
            best_accuracy = accuracy

            # Save model with the best accuracy
            save_model(linear, model_file_name)

    print(f'\nfile: {model_file_name}\naccuracy: {best_accuracy}\n')


if __name__ == '__main__':
    # Test Configurations
    debug_modes = [False, True]
    models = [LinearRegression, LogisticRegression]
    sources = ['pure.csv', 'group.csv']

    # Test each combination & save the best model
    for DEBUG in debug_modes:
        for m in models:
            for s in sources:
                save_best_model(m, s, TEST_ROUNDS, DEBUG)
