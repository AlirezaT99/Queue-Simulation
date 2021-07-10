import re
import numpy as np

from models import Queue

N = 10000
priority_prob = np.array([0.50, 0.20, 0.15, 0.10, 0.05])
fatigue_rate = -1


def sample_priorities(elements, n, weights) -> np.array:
    """returns an array of n weighted random samples with given probabilities"""
    return np.random.choice(elements, n, p=weights)


def process_input():
    """reads N+1 lines and reports the parameters"""
    global fatigue_rate
    input_line_1 = re.split(',\\s*|\\s+', input())
    parts_count, arrival_rate, reception_service_rate, fatigue_rate = map(float, input_line_1)
    queues = [re.split(',\\s*|\\s+', input()) for _ in range(int(parts_count))]
    queues = [[float(float(j)) for j in i] for i in queues]
    return arrival_rate, reception_service_rate, queues


def setup_queues(main_service, queues):
    """creates queues using the parameters provided"""
    for queue_info in queues:
        Queue(queue_info)
    Queue([main_service])  # Reception


def sample_arrivals(arrival_rate):
    inter_arrivals = np.random.exponential(arrival_rate, size=N)
    result = [inter_arrivals[0]]
    for i in range(1, N):
        result = np.append(result, result[-1] + inter_arrivals[i])
    return result


def run_simulation():
    arrival, service_rate, queues_info = process_input()
    setup_queues(service_rate, queues_info)
    customers_arrival = sample_arrivals(arrival)
    customers_priority = sample_priorities(range(5), N, priority_prob)


if __name__ == '__main__':
    run_simulation()
