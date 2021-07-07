import numpy as np


class Queue:
    queues = list()

    def __init__(self, service_rates):
        self.operators = [Operator(rate) for rate in service_rates]
        Queue.queues.append(self)


class Operator:
    def __init__(self, service_time):
        self.next_free = 0
        self.service_time = service_time
        self.service_log = np.array([])

    def is_available(self, time):
        return self.next_free <= time

    def assign_job(self, arrival, in_queue=False):
        service_time = np.random.exponential(self.service_time)
        self.service_log = np.append(self.service_log, service_time)
        if in_queue:
            self.next_free += service_time
        else:
            self.next_free = arrival + service_time
        return service_time

    def get_stats(self):
        return np.average(self.service_log), np.sum(self.service_log) / self.next_free
