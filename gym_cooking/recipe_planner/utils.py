from itertools import combinations
from collections import defaultdict

import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# PREDICATES
# --------------------------------------------------------------
class Predicate():
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __eq__(self, other):
        return (self.name == other.name) and (self.args == other.args)

    def __hash__(self):
        return hash((self.name, self.args))

    def __str__(self):
        return '{}({})'.format(self.name, ', '.join(self.args))

    def __copy__(self):
        return type(self)(self.args[0])


class NoPredicate(Predicate):
    def __init__(self, obj=None):
        self.name = None
        self.args = (None,)

    def __str__(self):
        return 'None'

class Fresh(Predicate):
    def __init__(self, obj):
        Predicate.__init__(self, 'Fresh', (obj,))

class Chopped(Predicate):
    def __init__(self, obj):
        Predicate.__init__(self, 'Chopped', (obj,))

class Cooked(Predicate):
    def __init__(self, obj):
        Predicate.__init__(self, 'Cooked', (obj,))

class Delivered(Predicate):
    def __init__(self, obj):
        Predicate.__init__(self, 'Delivered', (obj,))

class Merged(Predicate):
    def __init__(self, obj):
        Predicate.__init__(self, 'Merged', (obj,))




# ACTIONS
# -------------------------------------------------------------
class Action:
    def __init__(self, name, pre, post_add):
        self.name = name
        # pre, post_add, post_delete must be lists of Predicates
        self.pre = self.pre_default if (pre is None) else pre
        self.post_add = self.post_add_default if (post_add is None) else post_add
        # assume just delete all the preconditions
        self.set_specs()
        self.is_joint = False

    def __str__(self):
        return '{}({})'.format(self.name, ', '.join(self.args))

    def __repr__(self):
        return '{}({})'.format(self.name, ', '.join(self.args))

    def set_specs(self):
        self.specs = '\n{}({})\n'.format(self.name, ', '.join(self.args))
        self.specs += 'Preconditions: {}\n'.format(', '.join([str(_) for _ in self.pre]))
        post = ', '.join([str(_) for _ in self.post_add]) #+ ', !'.join([''] + [str(_) for _ in self.post_delete])
        self.specs += 'Postconditions: {}'.format(post)

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name) and (self.args == other.args)

    def __hash__(self):
        return hash((self.name, self.args))

    def is_valid_in(self, state):
        temp_state = copy.copy(state)
        for pre in self.pre:
            try:
                temp_state.delete_predicate(pre)
            except:
                return False
        return True

    def get_next_from(self, state):
        next_state = copy.copy(state)
        for predicate in self.pre:
            next_state.delete_predicate(predicate)  # remove first instance
        for predicate in self.post_add:
            next_state.add_predicate(predicate)
        return next_state

'''
Get(X)
Pre: None
Post: Fresh(X)
'''
class Get(Action):
    def __init__(self, obj, pre=None, post_add=None):
        self.args = (obj,)   #('Tomato')

        self.pre_default = [NoPredicate()]
        self.post_add_default = [Fresh(obj), NoPredicate()]

        Action.__init__(self, 'Get', pre, post_add)

'''
Chop(X)
Pre: Fresh(X)
Post: Chopped(X), !Fresh(X)
'''
class Chop(Action):
    def __init__(self, obj, pre=None, post_add=None):
        self.args = (obj,)

        self.pre_default = [Fresh(obj)]
        self.post_add_default = [Chopped(obj)]

        Action.__init__(self, 'Chop', pre, post_add)

'''
Merge(X, Y)
Pre: SomeState(X), SomeState(Y)
Post: Merged(X-Y), !SomeState(X), !SomeState(Y)
'''
class Merge(Action):
    def __init__(self, arg1, arg2, pre=None, post_add=None):
        self.args = (arg1, arg2)
        #self.args = tuple(sorted([arg1, arg2]))
        # sorted because it doesn't matter order of merging

        self.pre_default = [Chopped(arg1), Merged(arg2)]
        self.post_add_default = [Merged('-'.join(sorted(arg1.split('-') + arg2.split('-'))))]

        Action.__init__(self, 'Merge', pre, post_add)
'''
Deliver(X)
Pre: Plated(X)
Post: Delivered(X), !Plated(X)
'''
class Deliver(Action):
    def __init__(self, obj, pre=None, post_add=None):
        self.args = (obj,)
        self.pre_default = [Merged(obj)]
        self.post_add_default = [Delivered(obj)]
        Action.__init__(self, 'Deliver', pre, post_add)



# STRIPSSTATE
# --------------------------------------------------------------
class STRIPSState:
    def __init__(self):
        self.predicates = []   # can have multiple same predicates

    def __str__(self):
        return '[{}]'.format(', '.join([str(p) for p in self.predicates]))

    def __eq__(self, other):
        if other is None:
            return False
        return sorted([str(p) for p in self.predicates]) == sorted([str(p) for p in other.predicates])

    def __hash__(self):
        return hash(tuple(sorted([str(p) for p in self.predicates])))

    def __copy__(self):
        new = STRIPSState()
        new.predicates = [copy.copy(p) for p in self.predicates]
        return new

    def add_predicate(self, predicate):
        self.predicates.append(predicate)

    def delete_predicate(self, predicate):
        assert predicate in self.predicates, "{} not in this state".format(predicate)
        self.predicates.remove(predicate)   # remove first instance

    def contains(self, predicate):
        return predicate in self.predicates


# GRAPHING
# --------------------------------------------------------------
def make_predicate_graph(initial, action_path, draw=True):
    # generates graph where nodes are predicates, edges are actions
    #   pointing from preconditions to postconditions
    # initial = STRIPSState object
    # action_path = list of Action objects
    g = nx.DiGraph()
    node_labels = {}
    edge_labels = {}

    # add initial nodes
    for predicate in initial.predicates:
        g.add_node(predicate)#, label=str(predicate))
        node_labels[predicate] = 'START' if predicate==NoPredicate() else str(predicate)

    # for each action in the path, in order:
    for action in action_path:
        # all preconditions should already exist
        # set all preconditions to point to all new post conditions
        for post_add in action.post_add:
            g.add_node(post_add)#, label=str(post_add))
            node_labels[post_add] = str(post_add)
            for pre in action.pre:
                # prevent self loops on pots and None
                if (pre != post_add) and (post_add != Fresh('Pot')):
                    g.add_edge(pre, post_add)#, label=action['str_name'])
                    edge_labels[(pre, post_add)] = str(action)
    # delete NoPredicate node
    #g.remove_node(NoPredicate())
    # rename NoPredicate node
    predicate_graph = make_graph(g, node_labels, edge_labels, draw)
    return predicate_graph

def make_action_graph(initial, action_path, draw=True):
    # generates graph where nodes are actions, edges are predicates
    #   shared by precondition/postcondition of two sequential actions
    # action_path = list of Action objects
    g = nx.DiGraph()
    node_labels = {}
    edge_labels = {}

    for action in action_path:
        g.add_node(action, obj=action)
        node_labels[action] = str(action)

        # search for nodes whose postconditions match this nodes preconditions
        for other_action in g:   # other_action is a node
            for p in other_action.post_add:
                if p in action.pre and p != NoPredicate():
                    g.add_edge(other_action, action, obj=p)
                    edge_labels[(other_action, action)] = str(p)
    action_graph = make_graph(g, node_labels, edge_labels, draw)
    return action_graph


def make_graph(g, node_labels, edge_labels, draw):
    plt.figure()
    layout = nx.spring_layout(g)
    labels_layout = dict(map(lambda kv: (kv[0], kv[1]+np.asarray([0,-0.1])), layout.items()))
    nx.draw_networkx_nodes(g, pos=layout, node_size=100)
    nx.draw_networkx_labels(g, pos=labels_layout, labels=node_labels)
    nx.draw_networkx_edges(g, pos=layout)
    nx.draw_networkx_edge_labels(g, pos=layout, edge_labels=edge_labels)
    if draw:
        plt.show()
    return g


# --------------------------------------------------------------
def get_layers(recipe_tasks, initial):
    # given a set of recipe_tasks, sort into layers
    # initial is a STRIPSState object
    if len(recipe_tasks) == 0: return []

    free_tasks = []
    remaining_tasks = []
    next_initial = copy.copy(initial)
    for t in recipe_tasks:
        if t.is_valid_in(initial):
            free_tasks.append(t)
            for pre in t.pre:
                next_initial.delete_predicate(pre)
            for post in t.post_add:
                next_initial.add_predicate(post)
        else:
            remaining_tasks.append(t)

    return [free_tasks] + get_layers(remaining_tasks, next_initial)


