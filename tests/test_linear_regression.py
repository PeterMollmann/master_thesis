import numpy as np
from simpleRegression.fitFunctions import SimpleRegression


def test_linear_regression():
    # Example data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])
    z = 2 + 3 * x + 4 * y
    # Fit the data
    popt, _, _ = SimpleRegression(
        features=np.array([x, y]),
        target=z,
        degrees=[1, 1],
        use_interactions=False,
    )
    assert np.allclose(
        popt, [2, 3, 4], atol=0.1
    ), f"Expected parameters close to [2, 3, 4], got {popt}"


if __name__ == "__main__":
    test_linear_regression()
