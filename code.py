import numpy as np

N = 10000
customer_priority = np.array([0.50, 0.20, 0.15, 0.10, 0.05])


def sample_customers(elements, n, weights) -> np.array:
    """returns an array of n weighted randomly samples with given probabilities"""
    return np.random.choice(elements, n, p=weights)


if __name__ == '__main__':
    customers = sample_customers(range(5), N, customer_priority)
