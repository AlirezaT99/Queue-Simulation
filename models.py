from collections import defaultdict, deque

import numpy as np


class Queue:
    queues = list()

    def __init__(self, service_rates):
        self.queue = deque()
        self.operators = [Operator(rate) for rate in service_rates]
        Queue.queues.append(self)
        # TODO self.stats
        self.customer_wait = defaultdict(list)

    def run(self, arrivals, priorities):
        assert len(arrivals) == len(priorities)
        for i in range(len(arrivals)):
            # TODO check queue
            operator, err = self.next_free_operator(arrivals[i])
            if not err:
                operator.assign_job(arrivals[i], priorities[i])
                self.customer_wait[priorities[i]].append(0.0)
            else:  # TODO push to queue
                pass  # TODO pay attention to priorities

    def next_free_operator(self, time):
        """ either returns a free operator, if any.
        Or, returns the operator that will be free the earliest
        """
        for op in self.operators:
            if op.is_available(time):
                return op, False
        return min(self.operators, key=lambda k: k.next_free), True


class MainQueue(Queue):
    def run(self, arrivals, priorities):
        super().run(arrivals, priorities)
        # TODO set other queues' arrivals


class Operator:
    def __init__(self, service_time):
        self.next_free = 0
        self.service_time = service_time
        self.service_log = defaultdict(list)

    def is_available(self, time):
        return self.next_free <= time

    def assign_job(self, arrival, priority, in_queue=False):
        service_time = np.random.exponential(self.service_time)
        self.service_log[priority].append(service_time)
        if in_queue:
            self.next_free += service_time
        else:
            self.next_free = arrival + service_time
        return service_time

    def get_stats(self):
        return np.average(self.service_log), np.sum(self.service_log) / self.next_free
