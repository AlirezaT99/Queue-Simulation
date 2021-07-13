import re
from collections import defaultdict

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


def run_all_queues(customers_arrival, customers_priority, customers_early_departure, customers_next_queue):
    # Run reception queue
    main_queue: Queue = Queue.queues.pop()
    main_queue.run(customers_arrival, customers_priority, customers_early_departure, customers_next_queue)
    # Extract other queues' input
    queue_arrivals, queue_priority, queue_give_up = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(len(main_queue.next_queue)):
        queue_arrivals[main_queue.next_queue[i]] = main_queue.departure[i]
        queue_priority[main_queue.next_queue[i]] = main_queue.departed_priority[i]
        queue_give_up[main_queue.next_queue[i]] = main_queue.fatigue_remainder[i]
    # Run other queues
    for queue_idx in range(len(Queue.queues)):
        Queue.queues[queue_idx].run(queue_arrivals[queue_idx], queue_priority[queue_idx], queue_give_up[queue_idx])
    # Put main queue back in the list
    Queue.queues.insert(0, main_queue)


def display_stats():
    pass  # TODO


def run_simulation():
    # Setup
    queues_count, fatigue, arrival, service_rate, queues_info = process_input()
    setup_queues(service_rate, queues_info)
    # Sample
    customers_arrival = sample_arrivals(arrival, N)
    customers_priority = sample_priorities(range(5), N, priority_prob)
    customers_early_departure = sample_fatigue(fatigue, N)
    customers_next_queue = sample_next_queue(range(queues_count), N)
    # Run
    run_all_queues(customers_arrival, customers_priority, customers_early_departure, customers_next_queue)
    # Display stats
    display_stats()


if __name__ == '__main__':
    run_simulation()
