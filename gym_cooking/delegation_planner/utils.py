from collections import namedtuple
import numpy as np
import scipy as sp
import random
from utils.utils import agent_settings


class SubtaskAllocDistribution():
    """Represents a distribution over subtask allocations."""

    def __init__(self, subtask_allocs):
        # subtask_allocs are a list of tuples of (subtask, subtask_agents).
        self.probs = {}
        if len(subtask_allocs) == 0:
            return
        prior = 1./(len(subtask_allocs))
        print('set prior', prior)

        for subtask_alloc in subtask_allocs:
            self.probs[tuple(subtask_alloc)] = prior

    def __str__(self):
        s = ''
        for subtask_alloc, p in self.probs.items():
            s += str(subtask_alloc) + ': ' + str(p) + '\n'
        return s

    def enumerate_subtask_allocs(self):
        return list(self.probs.keys())

    def get_list(self):
        return list(self.probs.items())

    def get(self, subtask_alloc):
        return self.probs[tuple(subtask_alloc)]

    def get_max(self):
        if len(self.probs) > 0:
            max_prob = max(self.probs.values())
            max_subtask_allocs = [subtask_alloc for subtask_alloc, p in self.probs.items() if p == max_prob]
            return random.choice(max_subtask_allocs)
        return None

    def get_max_bucketed(self):
        subtasks = []
        probs = []
        for subtask_alloc, p in self.probs.items():
            for t in subtask_alloc:
                if agent_name in t.subtask_agent_names:
                    # If already accounted for, then add probability.
                    if t in subtasks:
                        probs[subtasks.index(t)] += p
                    # Otherwise, make a new element in the distribution.
                    else:
                        subtasks.append(t)
                        probs.append(p)
        best_subtask = subtasks[np.argmax(probs)]
        return self.probs.get_best_containing(best_subtask)

    def get_best_containing(self, subtask):
        """Return max likelihood subtask_alloc that contains the given subtask."""
        valid_subtask_allocs = []
        valid_p = []
        for subtask_alloc, p in self.probs.items():
            if subtask in subtask_alloc:
                valid_subtask_allocs.append(subtask)
                valid_p.append(p)
        return valid_subtask_allocs[np.argmax(valid_p)]

    def set(self, subtask_alloc, value):
        self.probs[tuple(subtask_alloc)] = value

    def update(self, subtask_alloc, factor):
        self.probs[tuple(subtask_alloc)] *= factor

    def delete(self, subtask_alloc):
        try:
            del self.probs[tuple(subtask_alloc)]
        except:
            print('subtask_alloc {} not found in probsdict'.format(subtask_alloc))

    def normalize(self):
        total = sum(self.probs.values())
        for subtask_alloc in self.probs.keys():
            if total == 0:
                self.probs[subtask_alloc] = 1./len(self.probs)
            else:
                self.probs[subtask_alloc] *= 1./total
        return self.probs
