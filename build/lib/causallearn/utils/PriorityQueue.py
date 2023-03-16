"""
Code modified from:
https://github.com/jmschrei/pomegranate/blob/master/pomegranate/utils.pyx
"""
import heapq


class PriorityQueue:
    def __init__(self):
        self.n = 0
        self.pq = []
        self.entries = {}

    def __len__(self):
        return self.n

    def push(self, item, weight):
        entry = [weight, item]
        self.entries[item[0]] = entry
        heapq.heappush(self.pq, entry)
        self.n += 1

    def get(self, variables):
        return self.entries.get(variables, None)

    def delete(self, variables):
        entry = self.entries.pop(variables)
        entry[-1] = ((-1,),)
        self.n -= 1

    def empty(self):
        return self.n == 0

    def pop(self):
        while not self.empty():
            weight, item = heapq.heappop(self.pq)
            if item[0] != (-1,):
                del self.entries[item[0]]
                self.n -= 1
                return weight, item
        else:
            raise KeyError("Attempting to pop from an empty priority queue")
