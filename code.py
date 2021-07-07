import re
import numpy as np

from models import Queue

N = 10000
customer_priority = np.array([0.50, 0.20, 0.15, 0.10, 0.05])


def sample_customers(elements, n, weights) -> np.array:
    """returns an array of n weighted random samples with given probabilities"""
    return np.random.choice(elements, n, p=weights)


def process_input():
    """reads N+1 lines and creates the queues accordingly"""
    input_line_1 = re.split(',\\s*|\\s+', input())
    parts_count, arrival_rate, reception_service_rate, fatigue_rate = input_line_1
    for i in range(parts_count):
        service_rates = re.split(',\\s*|\\s+', input())
        Queue(service_rates)


if __name__ == '__main__':
    process_input()
    customers_priorities = sample_customers(range(5), N, customer_priority)
