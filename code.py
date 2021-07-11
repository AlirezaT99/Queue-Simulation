import re
import numpy as np

from models import Queue

N = 10000
priority_prob = np.array([0.50, 0.20, 0.15, 0.10, 0.05])
total_wait = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def sample_priorities(elements, n, weights) -> np.array:
    """returns an array of n weighted random samples with given probabilities"""
    return np.random.choice(elements, n, p=weights)


def sample_next_queue(elements, n) -> np.array:
    """returns an array of n random samples from given elements"""
    return np.random.choice(elements, n)


def sample_arrivals(arrival_rate, n):
    inter_arrivals = np.random.exponential(1 / arrival_rate, size=n)
    result = [inter_arrivals[0]]
    for i in range(1, n):
        result = np.append(result, result[-1] + inter_arrivals[i])
    return result


def process_input():
    """reads N+1 lines and reports the parameters"""
    input_line_1 = map(float, re.split(',\\s*|\\s+', input()))
    parts_count, arrival_rate, reception_service_rate, fatigue_rate = input_line_1
    queues = [list(map(float, re.split(',\\s*|\\s+', input()))) for _ in range(int(parts_count))]
    return int(parts_count), fatigue_rate, arrival_rate, reception_service_rate, queues


def setup_queues(main_service, queues):
    """creates queues using the parameters provided"""
    for queue_info in queues:
        Queue(queue_info)
    Queue([main_service])  # Reception


def sample_fatigue(fatigue, n):
    return np.random.exponential(1 / fatigue, size=n)


def run_simulation():
    # Setup
    queues_count, fatigue, arrival, service_rate, queues_info = process_input()
    setup_queues(service_rate, queues_info)
    # Sample
    customers_arrival = sample_arrivals(arrival, N)
    customers_priority = sample_priorities(range(5), N, priority_prob)
    customers_next_queue = sample_next_queue(range(queues_count), N)
    customers_early_departure = sample_fatigue(fatigue, N)
    # Run
    main_queue = Queue.queues.pop()
    main_queue.run(customers_arrival, customers_priority)
    for queue in Queue.queues:
        queue.run([], [])  # TODO set by main
    # Display stats
    pass


if __name__ == '__main__':
    run_simulation()
