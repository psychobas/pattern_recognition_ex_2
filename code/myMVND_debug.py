import numpy as np

class MVND:
    # TODO: EXERCISE 2 - Implement mean and covariance matrix of given data
    def __init__(self, data: np.ndarray, c: float = 1.0):
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        self.mean = np.mean(self.data, axis = 1)
        print("mean is: ", self.mean)
        self.cov  = np.cov(self.data)
        print("cov is: ", self.cov)

    # TODO: EXERCISE 2 - Implement pdf and logpdf of a MVND
    def pdf(self, x: np.ndarray) -> np.ndarray:       # Alternatively a float can also be returned if individual datapoints are computed
        print("x is: ", x)
        print(x.shape == self.mean.shape)
        """pdf of the multivariate normal distribution."""
        x_centered = x - self.mean
        print("x_centered is: ", x_centered)

        d = self.cov.ndim
        print("d is: ", d)
        pdf = (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))) * np.exp(-(np.linalg.solve(self.cov, x_centered).T.dot(x_centered)) / 2))

        return pdf



test_data = np.random.rand(2,20)

test = MVND(data = test_data)

print(test.pdf(x = np.random.rand(2,1)))