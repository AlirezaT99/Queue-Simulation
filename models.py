from collections import defaultdict, deque
import numpy as np


class Queue:
    queues = list()
    customer_in_system = defaultdict(int)

    def __init__(self, service_rates):
        self.queue = deque()
        self.operators = [Operator(rate) for rate in service_rates]
        Queue.queues.append(self)
        self.customer_wait = defaultdict(list)
        self.early_departed = 0
        self.queue_len = list()
        # input for next queues
        self.departure = list()
        self.fatigue_remainder = list()
        self.departed_priority = list()
        self.next_queue = list()

    def run(self, arrivals, priorities, give_up_times, next_queue=None):
        assert len(arrivals) == len(priorities)
        for i in range(len(arrivals)):
            self.queue_len.append(len(self.queue))
            next_op = self.next_free_operator(arrivals[i])
            customer_next_queue = next_queue[i] if next_queue is not None else None
            # Who else is exhausted ?
            for customer in self.queue:
                if arrivals[i] >= customer[1] + customer[3]:
                    self.customer_wait[customer[2]].append(customer[3])
                    self.early_departed += 1
            self.queue = deque(filter(lambda c: arrivals[i] < c[1] + c[3], self.queue))
            # Handle queue (if there exists a free op and someone in the queue)
            while self.queue and (not next_op[1]):
                customer = self.queue.popleft()  # (idx, arrival, priority, give_up, next_queue)
                free_time = next_op[0].next_free
                self.assign_to_operator(next_op[0], customer[1], customer[2], customer[3], customer[4], in_queue=True)
                self.customer_wait[customer[2]].append(free_time - customer[1])
                next_op = self.next_free_operator(arrivals[i])
            # Handle the new arrival
            operator, err = self.next_free_operator(arrivals[i])
            if not err:  # No need to wait
                self.assign_to_operator(operator, arrivals[i], priorities[i], give_up_times[i], customer_next_queue)
                self.customer_wait[priorities[i]].append(0)
            else:  # wait in queue
                self.push_to_queue(i, arrivals[i], priorities[i], give_up_times[i], customer_next_queue)
        # After arrivals, some customers might still be in the queue
        next_op = self.next_free_operator(self.departure[-1])
        while self.queue:
            self.queue_len.append(len(self.queue))
            customer = self.queue.popleft()
            free_time = next_op[0].next_free
            self.assign_to_operator(next_op[0], customer[1], customer[2], customer[3], customer[4], in_queue=True)
            self.customer_wait[customer[2]].append(free_time - customer[1])
            next_op = self.next_free_operator(self.departure[-1])

    def assign_to_operator(self, operator, arrival, priority, give_up_time, customer_next_queue, in_queue=False):
        old_next_free = operator.next_free
        operator.assign_job(arrival, priority, in_queue)
        # What if the customer decides to leave during service
        if operator.next_free > arrival + give_up_time:
            operator.next_free = arrival + give_up_time
            operator.service_log[priority].pop()  # Chose not to calculate their service
            self.customer_wait[priority].append(give_up_time)
            self.early_departed += 1
        else:  # successful departure
            self.departure.append(operator.next_free)
            self.departed_priority.append(priority)
            self.fatigue_remainder.append(give_up_time - (operator.next_free - arrival))
            if customer_next_queue is not None:  # only used for main queue
                self.next_queue.append(customer_next_queue)
        # add presence to every time unit
        for t in range(int(arrival), int(operator.next_free) + 1):
            Queue.customer_in_system[t / 60] += 1  # x-axis is minutes

    def next_free_operator(self, time):
        """ either returns a free operator, if any.
        Or, returns the operator that will be free the earliest
        """
        for op in self.operators:
            if op.is_available(time):
                return op, False
        return min(self.operators, key=lambda k: k.next_free), True

    def push_to_queue(self, idx, arrival_time, priority, give_up_time, customer_next_queue):
        """pushes customers to queue.
        The new customer is prior to old ones if has a higher priority.
        Tie breaker: The one who arrived sooner
        TODO O(n) -> O(log(n)) using binary search
        """
        new_customer = (idx, arrival_time, priority, give_up_time, customer_next_queue)
        if not self.queue:
            self.queue.appendleft(new_customer)
        else:
            for i in range(len(self.queue)):
                customer_i = self.queue[i]
                if (new_customer[2], -new_customer[1]) > (customer_i[2], -customer_i[1]):
                    self.queue.insert(i, new_customer)
                    break
            else:
                self.queue.append(new_customer)  # Worst case: not prior to anyone


class Operator:
    def __init__(self, service_time):
        self.next_free = 0
        self.service_time = service_time
        self.service_log = defaultdict(list)

    def is_available(self, time):
        return self.next_free <= time

    def assign_job(self, arrival, priority, in_queue=False):
        service_time = np.random.exponential(1 / self.service_time)
        self.service_log[priority].append(service_time)
        if in_queue:
            self.next_free += service_time
        else:
            self.next_free = arrival + service_time
        return service_time

    def get_stats(self):
        return np.average(self.service_log), np.sum(self.service_log) / self.next_free
