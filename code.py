import re
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from models import Queue

N = int(1e7)
priority_prob = np.array([0.50, 0.20, 0.15, 0.10, 0.05])


def sample_priorities(elements, n, weights) -> np.array:
    """returns an array of n weighted random samples with given probabilities"""
    return np.random.choice(elements, n, p=weights)


def sample_next_queue(elements, n) -> np.array:
    """returns an array of n random samples from given elements"""
    return np.random.choice(elements, n)


def sample_arrivals(arrival_rate, n):
    """samples n arrivals using exponential inter-arrival samples"""
    inter_arrivals = np.random.exponential(1 / arrival_rate, size=n)
    result = [inter_arrivals[0]]
    for i in range(1, n):
        result.append(result[-1] + inter_arrivals[i])
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


def run_all_queues(customers_arrival, customers_priority, customers_early_departure, customers_next_queue, user_dict):
    # Run reception queue
    main_queue: Queue = Queue.queues.pop()
    main_queue.run(customers_arrival, customers_priority, customers_early_departure, user_dict, list(user_dict.keys()),
                   customers_next_queue)
    # Extract other queues' input
    queue_arrivals, queue_priority, queue_give_up, queue_users = defaultdict(list), defaultdict(list), defaultdict(
        list), defaultdict(list)
    for i in range(len(main_queue.next_queue)):
        queue_arrivals[main_queue.next_queue[i]].append(main_queue.departure[i])
        queue_priority[main_queue.next_queue[i]].append(main_queue.departed_priority[i])
        queue_give_up[main_queue.next_queue[i]].append(main_queue.fatigue_remainder[i])
        queue_users[main_queue.next_queue[i]].append(main_queue.next_queue_users[i])
    # Run other queues
    for queue_idx in range(len(Queue.queues)):
        Queue.queues[queue_idx].run(queue_arrivals[queue_idx], queue_priority[queue_idx], queue_give_up[queue_idx],
                                    user_dict, queue_users[queue_idx])
    # Put main queue back in the list
    Queue.queues.insert(0, main_queue)


def display_stats(user_dict):
    """TODO check if np.sum performs better"""
    queue_wait = dict()
    service_time = dict()
    for i in range(5):  # priorities
        level_i_count = sum([len(queue.customer_wait[i]) for queue in Queue.queues])
        level_i_wait = sum([sum(queue.customer_wait[i]) for queue in Queue.queues])
        queue_wait[i] = (level_i_count, level_i_wait)
        # service_time[i] = sum([sum(op.service_log[i]) for op in queue.operators] for queue in Queue.queues)
        total_service_i = 0
        for queue in Queue.queues:
            for op in queue.operators:
                total_service_i += sum(op.service_log[i])
        service_time[i] = total_service_i

    print('1.\tAverage time spent in system by customers:')
    print(f'\t\t- All: {sum([queue_wait[i][1] + service_time[i] for i in range(5)]) / N}')
    for i in range(4, -1, -1):
        print(f'\t\t- Priority {i}: {(queue_wait[i][1] + service_time[i]) / queue_wait[i][0]}')
    #
    print('2.\tAverage time spent in queues by customers:')
    print(f'\t\t- All: {sum([queue_wait[i][1] for i in range(5)]) / N}')
    for i in range(4, -1, -1):
        print(f'\t\t- Priority {i}: {queue_wait[i][1] / queue_wait[i][0]}')
    #
    total_early_departed = sum([queue.early_departed for queue in Queue.queues])
    print(f'3.\tTotal {total_early_departed} got exhausted and left early.')
    #
    print(f'4.\tAverage length of every queue in system:')
    print(f'\t\t- Main Queue: {np.average(Queue.queues[0].queue_len)}')
    for i in range(len(Queue.queues) - 1):
        print(f'\t\t- Queue {i + 1}: {np.average(Queue.queues[i].queue_len)}')
    #
    for queue_idx in range(len(Queue.queues)):
        plt.plot(Queue.queues[queue_idx].queue_len)
        plt.title(f'6.{queue_idx}. Queue {queue_idx}' if queue_idx > 0 else 'Main Queue')
        plt.show()
    #
    plt.plot(sorted(Queue.customer_in_system.keys()), Queue.customer_in_system.values())
    plt.title(f'7. Number of customers in system per time')
    plt.show()

    plt.plot(sorted(user_dict.keys()), user_dict.values())
    plt.title(f'7. Service Time')
    plt.show()
    # TODO 9
    # TODO 10
    # TODO 5


def run_simulation():
    # Setup
    queues_count, fatigue, arrival, service_rate, queues_info = process_input()
    setup_queues(service_rate, queues_info)
    # Sample
    customers_arrival = sample_arrivals(arrival, N)
    customers_priority = sample_priorities(range(5), N, priority_prob)
    customers_early_departure = sample_fatigue(fatigue, N)
    customers_next_queue = sample_next_queue(range(queues_count), N)
    user_dict = {i: 0 for i in range(N)}
    # Run
    start = time.time()
    run_all_queues(customers_arrival, customers_priority, customers_early_departure, customers_next_queue, user_dict)
    print(f'>>> running queues took {time.time() - start}')
    # Display stats
    start = time.time()
    display_stats(user_dict)
    print(f'>>> gathering stats took {time.time() - start}')


if __name__ == '__main__':
    run_simulation()
