import re
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from models import Queue

N = int(1e7)
priority_prob = np.array([0.50, 0.20, 0.15, 0.10, 0.05])
arrival, fatigue = -1, -1


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
    # Reset Queue
    Queue.queues.clear()
    Queue.customer_in_system = defaultdict(int)

    for queue_info in queues:
        Queue(queue_info)
    Queue([main_service])  # Reception


def sample_fatigue(fatigue_rate, n):
    return np.random.exponential(1 / fatigue_rate, size=n)


def run_all_queues(customers_arrival, customers_priority, customers_early_departure, customers_next_queue, total_wait):
    # Run reception queue
    main_queue: Queue = Queue.queues.pop()
    main_queue.run(customers_arrival, customers_priority, customers_early_departure, total_wait,
                   list(total_wait.keys()), customers_next_queue)
    # Extract other queues' input
    queue_arrivals, queue_priority = defaultdict(list), defaultdict(list)
    queue_give_up, queue_customers = defaultdict(list), defaultdict(list)
    for i in range(len(main_queue.next_queue)):
        queue_arrivals[main_queue.next_queue[i]].append(main_queue.departure[i])
        queue_priority[main_queue.next_queue[i]].append(main_queue.departed_priority[i])
        queue_give_up[main_queue.next_queue[i]].append(main_queue.fatigue_remainder[i])
        queue_customers[main_queue.next_queue[i]].append(main_queue.next_queue_customers[i])
    # Run other queues
    for queue_idx in range(len(Queue.queues)):
        Queue.queues[queue_idx].run(queue_arrivals[queue_idx], queue_priority[queue_idx], queue_give_up[queue_idx],
                                    total_wait, queue_customers[queue_idx])
    # Put main queue back in the list
    Queue.queues.insert(0, main_queue)


def display_stats(customer_time, customers_priority: np.array):
    queue_wait, service_time = defaultdict(list), defaultdict(list)
    sum_queue_wait, sum_service_time = defaultdict(int), defaultdict(int)
    priority_count, overall_time = defaultdict(), defaultdict()
    for i in range(len(customers_priority)):
        queue_wait[customers_priority[i]].append(customer_time[i]['queue'])
        sum_queue_wait[customers_priority[i]] += customer_time[i]['queue']
        service_time[customers_priority[i]].append(customer_time[i]['service'])
        sum_service_time[customers_priority[i]] += customer_time[i]['service']
    for i in range(5):
        priority_count[i] = np.sum(customers_priority == i)
        overall_time[i] = sum_queue_wait[i] + sum_service_time[i]
    print('1.\tAverage time spent in system by customers:')
    print(f'\t\t- All: {round(sum(overall_time.values()) / N, 2)}s')
    for i in range(4, -1, -1):
        print(f'\t\t- Priority {i}: {round(overall_time[i] / priority_count[i], 2)}s')
    #
    print('2.\tAverage time spent in queues by customers:')
    print(f'\t\t- All: {round(sum(sum_queue_wait.values()) / N, 2)}s')
    for i in range(4, -1, -1):
        print(f'\t\t- Priority {i}: {round(sum_queue_wait[i] / priority_count[i], 2)}s')
    #
    total_early_departed = sum([queue.early_departed for queue in Queue.queues])
    print(f'3.\tTotal {total_early_departed} got exhausted and left early.')
    #
    print(f'4.\tAverage length of every queue in system:')
    print(f'\t\t- Main Queue: {round(np.average(Queue.queues[0].queue_len), 2)}')
    for i in range(1, len(Queue.queues)):
        print(f'\t\t- Queue {i}: {round(np.average(Queue.queues[i].queue_len), 2)}')
    #
    for queue_idx in range(len(Queue.queues)):
        plt.plot(Queue.queues[queue_idx].queue_len)
        plot_title = f'Queue {queue_idx}' if queue_idx > 0 else 'Main Queue'
        plt.title(f'6.{queue_idx}. Length of {plot_title} per time')
        plt.show()
    #
    sorted_count = sorted(Queue.customer_in_system.items(), key=lambda tpl: tpl[0])
    plt.plot(*zip(*sorted_count))
    plt.title(f'7. Number of customers in system per time')
    plt.show()
    #
    for i in range(5):
        plt.hist(service_time[i])
        plt.title(f'9.{i} Total service time (Priority {i})')
        plt.show()
    #
    for i in range(5):
        plt.hist(queue_wait[i])
        plt.title(f'10.{i} Total waiting time (Priority {i})')
        plt.show()
    #
    rate = find_appropriate_rate()
    print(f'5. The mean service rate that causes zero customer in queues: {rate}')


def find_appropriate_rate(epochs=20, high_precision=False):
    # run gc for no apparent reason
    import gc
    gc.collect()
    # Trade-off between time and accuracy
    global N
    if not high_precision:
        N = int(1e4)
    best_rate = -1
    queues_count = len(Queue.queues) - 1
    main_service_rate = Queue.queues[0].operators[0].service_rate
    mean_queue_ops = int(np.mean([len(queue.operators) for queue in Queue.queues[1:]]))
    step = new_rate = np.mean([np.mean([op.service_rate for op in queue.operators]) for queue in Queue.queues[1:]])
    going_backwards = False
    for i in range(epochs):
        if are_queues_empty():
            best_rate = new_rate
            step /= 2
            new_rate -= step
            going_backwards = True
        else:
            if going_backwards:
                step /= 2
                going_backwards = False
            else:
                step *= 2
            new_rate += step
        queues_info = [[new_rate] * mean_queue_ops] * queues_count
        # print(f'\tAttempt No.{i + 1} with mean rate {new_rate}')
        run_simulation((queues_count, fatigue, arrival, main_service_rate, queues_info))

    return best_rate


def are_queues_empty():
    return sum([sum(queue.queue_len) for queue in Queue.queues[1:]]) == 0


def run_simulation(params=None):
    global fatigue, arrival
    # Setup
    if params is None:
        queues_count, fatigue, arrival, service_rate, queues_info = process_input()
    else:
        queues_count, fatigue, arrival, service_rate, queues_info = params
    setup_queues(service_rate, queues_info)
    total_time = {i: {'queue': 0, 'service': 0} for i in range(N)}
    # Sample
    start = time.time()
    customers_arrival = sample_arrivals(arrival, N)
    customers_priority = sample_priorities(range(5), N, priority_prob)
    customers_early_departure = sample_fatigue(fatigue, N)
    customers_next_queue = sample_next_queue(range(queues_count), N)
    if params is None:
        print(f'>>> sampling took {round(time.time() - start, 3)}s')
    # Run
    start = time.time()
    run_all_queues(customers_arrival, customers_priority, customers_early_departure, customers_next_queue, total_time)
    if params is None:
        print(f'>>> running queues took {round(time.time() - start, 3)}s')
    # Display stats
    if params is None:
        display_stats(total_time, customers_priority)


if __name__ == '__main__':
    run_simulation()
