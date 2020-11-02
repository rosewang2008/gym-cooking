class Delegator:
    def __init__(self):
        pass
    def delegate(self, agents, tasks):
        raise NotImplementedError()
    def assign(self, agents, delegated_tasks):
        for i, a in enumerate(agents):
            a.tasks = delegated_tasks[i]
