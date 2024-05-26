__all__ = [
    'features', 'target', 'TEST_SIZE', 'state', 'TEST_ROUNDS', 'generate_k'
]

# Define columns to use for training
"""
w: Weight ( kg )
h: Height ( cm )
k: Generated ( w * 0.02 + h * 0.06 - 10.29 )
"""
features = ['w', 'h', 'k']
target = 'gender'

# Test Size
TEST_SIZE = 0.1


def state(debug: bool = False) -> int | None:
    """
	Generate random state based on debug mode
	:param debug: bool
	:return: random_state as int
	"""
    return 50 if debug else None


def generate_k(w: float, h: float) -> float:
    return w * 0.02 + h * 0.06 - 10.29


# Test rounds to be run per each test combination
TEST_ROUNDS = 30
