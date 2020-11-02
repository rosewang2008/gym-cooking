import numpy as np
from collections import defaultdict, OrderedDict
from itertools import product, combinations
import networkx as nx
import copy
import matplotlib.pyplot as plt
from functools import lru_cache

import recipe_planner.utils as recipe
from navigation_planner.utils import manhattan_dist
from utils.core import Object, GridSquare, Counter


class World:
    """World class that hold all of the non-agent objects in the environment."""
    NAV_ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, arglist):
        self.rep = [] # [row0, row1, ..., rown]
        self.arglist = arglist
        self.objects = defaultdict(lambda : [])

    def get_repr(self):
        return self.get_dynamic_objects()

    def __str__(self):
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __copy__(self):
        new = World(self.arglist)
        new.__dict__ = self.__dict__.copy()
        new.objects = copy.deepcopy(self.objects)
        new.reachability_graph = self.reachability_graph
        new.distances = self.distances
        return new

    def update_display(self):
        # Reset the current display (self.rep).
        self.rep = [[' ' for i in range(self.width)] for j in range(self.height)]
        objs = []
        for o in self.objects.values():
            objs += o
        for obj in objs:
            self.add_object(obj, obj.location)
        for obj in self.objects["Tomato"]:
            self.add_object(obj, obj.location)
        return self.rep

    def print_objects(self):
        for k, v in self.objects.items():
            print(k, list(map(lambda o: o.location, v)))

    def make_loc_to_gridsquare(self):
        """Creates a mapping between object location and object."""
        self.loc_to_gridsquare = {}
        for obj in self.get_object_list():
            if isinstance(obj, GridSquare):
                self.loc_to_gridsquare[obj.location] = obj

    def make_reachability_graph(self):
        """Create a reachability graph between world objects."""
        self.reachability_graph = nx.Graph()
        for x in range(self.width):
            for y in range(self.height):
                location = (x, y)
                gs = self.loc_to_gridsquare[(x, y)]

                # If not collidable, add node with direction (0, 0).
                if not gs.collidable:
                    self.reachability_graph.add_node((location, (0, 0)))

                # Add nodes for collidable gs + all edges.
                for nav_action in World.NAV_ACTIONS:
                    new_location = self.inbounds(location=tuple(np.asarray(location) + np.asarray(nav_action)))
                    new_gs = self.loc_to_gridsquare[new_location]

                    # If collidable, add edges for adjacent noncollidables.
                    if gs.collidable and not new_gs.collidable:
                        self.reachability_graph.add_node((location, nav_action))
                        if (new_location, (0, 0)) in self.reachability_graph:
                            self.reachability_graph.add_edge((location, nav_action),
                                                             (new_location, (0, 0)))
                    # If not collidable and new_gs collidable, add edge.
                    elif not gs.collidable and new_gs.collidable:
                        if (new_location, tuple(-np.asarray(nav_action))) in self.reachability_graph:
                            self.reachability_graph.add_edge((location, (0, 0)),
                                                             (new_location, tuple(-np.asarray(nav_action))))
                    # If both not collidable, add direct edge.
                    elif not gs.collidable and not new_gs.collidable:
                        if (new_location, (0, 0)) in self.reachability_graph:
                            self.reachability_graph.add_edge((location, (0, 0)), (new_location, (0, 0)))
                    # If both collidable, add nothing.

        # If you want to visualize this graph, uncomment below.
        # plt.figure()
        # nx.draw(self.reachability_graph)
        # plt.show()

    def get_lower_bound_between(self, subtask, agent_locs, A_locs, B_locs):
        """Return distance lower bound between subtask-relevant locations."""
        lower_bound = self.perimeter + 1
        for A_loc, B_loc in product(A_locs, B_locs):
            bound = self.get_lower_bound_between_helper(
                    subtask=subtask,
                    agent_locs=agent_locs,
                    A_loc=A_loc,
                    B_loc=B_loc)
            if bound < lower_bound:
                lower_bound = bound
        return lower_bound

    @lru_cache(maxsize=40000)
    def get_lower_bound_between_helper(self, subtask, agent_locs, A_loc, B_loc):
        lower_bound = self.perimeter + 1
        A = self.get_gridsquare_at(A_loc)
        B = self.get_gridsquare_at(B_loc)
        A_possible_na = [(0, 0)] if not A.collidable else World.NAV_ACTIONS
        B_possible_na = [(0, 0)] if not B.collidable else World.NAV_ACTIONS

        for A_na, B_na in product(A_possible_na, B_possible_na):
            if len(agent_locs) == 1:
                try:
                    bound_1 = nx.shortest_path_length(
                            self.reachability_graph, (agent_locs[0], (0, 0)), (A_loc, A_na))
                    bound_2 = nx.shortest_path_length(
                            self.reachability_graph, (A_loc, A_na), (B_loc, B_na))
                except:
                    continue
                bound = bound_1 + bound_2 - 1

            elif len(agent_locs) == 2:
                # Try to calculate the distances between agents and Objects A and B.
                # Distance between Agent 1 <> Object A.
                try:
                    bound_1_to_A = nx.shortest_path_length(
                            self.reachability_graph, (agent_locs[0], (0, 0)), (A_loc, A_na))
                except:
                    bound_1_to_A = self.perimeter
                # Distance between Agent 2 <> Object A.
                try:
                    bound_2_to_A = nx.shortest_path_length(
                            self.reachability_graph, (agent_locs[1], (0, 0)), (A_loc, A_na))
                except:
                    bound_2_to_A = self.perimeter

                # Take the agent that's the closest to Object A.
                min_bound_to_A = min(bound_1_to_A, bound_2_to_A)

                # Distance between the agents.
                bound_between_agents = manhattan_dist(A_loc, B_loc)

                # Distance between Agent 1 <> Object B.
                try:
                    bound_1_to_B = nx.shortest_path_length(self.reachability_graph, (agent_locs[0], (0, 0)), (B_loc, B_na))
                except:
                    bound_1_to_B = self.perimeter

                # Distance between Agent 2 <> Object B.
                try:
                    bound_2_to_B = nx.shortest_path_length(self.reachability_graph, (agent_locs[1], (0, 0)), (B_loc, B_na))
                except:
                    bound_2_to_B = self.perimeter

                # Take the agent that's the closest to Object B.
                min_bound_to_B = min(bound_1_to_B, bound_2_to_B)

                # For chop or deliver, must bring A to B.
                if isinstance(subtask, recipe.Chop) or isinstance(subtask, recipe.Deliver):
                    bound = min_bound_to_A + bound_between_agents - 1
                # For merge, agents can separately go to A and B and then meet in the middle.
                elif isinstance(subtask, recipe.Merge):
                    min_bound_to_A, min_bound_to_B = self.check_bound(
                            min_bound_to_A=min_bound_to_A,
                            min_bound_to_B=min_bound_to_B,
                            bound_1_to_A=bound_1_to_A,
                            bound_2_to_A=bound_2_to_A,
                            bound_1_to_B=bound_1_to_B,
                            bound_2_to_B=bound_2_to_B
                            )
                    bound = max(min_bound_to_A, min_bound_to_B) + (bound_between_agents - 1)/2

            if bound < lower_bound:
                lower_bound = bound

        return max(1, lower_bound)

    def check_bound(self, min_bound_to_A, min_bound_to_B,
                            bound_1_to_A, bound_2_to_A,
                            bound_1_to_B, bound_2_to_B):
        # Checking for whether it's the same agent that does the subtask.
        if ((bound_1_to_A == min_bound_to_A and bound_1_to_B == min_bound_to_B) or
            (bound_2_to_A == min_bound_to_A and bound_2_to_B == min_bound_to_B)):
            return 2*min_bound_to_A, 2*min_bound_to_B
        return min_bound_to_A, min_bound_to_B

    def is_occupied(self, location):
        o = list(filter(lambda obj: obj.location == location and
         isinstance(obj, Object) and not(obj.is_held), self.get_object_list()))
        if o:
            return True
        return False

    def clear_object(self, position):
        """Clears object @ position in self.rep and replaces it with an empty space"""
        x, y = position
        self.rep[y][x] = ' '

    def clear_all(self):
        self.rep = []

    def add_object(self, object_, position):
        x, y = position
        self.rep[y][x] = str(object_)

    def insert(self, obj):
        self.objects.setdefault(obj.name, []).append(obj)

    def remove(self, obj):
        num_objs = len(self.objects[obj.name])
        index = None
        for i in range(num_objs):
            if self.objects[obj.name][i].location == obj.location:
                index = i
        assert index is not None, "Could not find {}!".format(obj.name)
        self.objects[obj.name].pop(index)
        assert len(self.objects[obj.name]) < num_objs, "Nothing from {} was removed from world.objects".format(obj.name)

    def get_object_list(self):
        all_obs = []
        for o in self.objects.values():
            all_obs += o
        return all_obs

    def get_dynamic_objects(self):
        """Get objects that can be moved."""
        objs = list()

        for key in sorted(self.objects.keys()):
            if key != "Counter" and key != "Floor" and "Supply" not in key and key != "Delivery" and key != "Cutboard":
                objs.append(tuple(list(map(lambda o: o.get_repr(), self.objects[key]))))

        # Must return a tuple because this is going to get hashed.
        return tuple(objs)

    def get_collidable_objects(self):
        return list(filter(lambda o : o.collidable, self.get_object_list()))

    def get_collidable_object_locations(self):
        return list(map(lambda o: o.location, self.get_collidable_objects()))

    def get_dynamic_object_locations(self):
        return list(map(lambda o: o.location, self.get_dynamic_objects()))

    def is_collidable(self, location):
        return location in list(map(lambda o: o.location, list(filter(lambda o: o.collidable, self.get_object_list()))))

    def get_object_locs(self, obj, is_held):
        if obj.name not in self.objects.keys():
            return []

        if isinstance(obj, Object):
            return list(
                    map(lambda o: o.location, list(filter(lambda o: obj == o and
                    o.is_held == is_held, self.objects[obj.name]))))
        else:
            return list(map(lambda o: o.location, list(filter(lambda o: obj == o,
                self.objects[obj.name]))))

    def get_all_object_locs(self, obj):
        return list(set(self.get_object_locs(obj=obj, is_held=True) + self.get_object_locs(obj=obj, is_held=False)))

    def get_object_at(self, location, desired_obj, find_held_objects):
        # Map obj => location => filter by location => return that object.
        all_objs = self.get_object_list()

        if desired_obj is None:
            objs = list(filter(
                lambda obj: obj.location == location and isinstance(obj, Object) and obj.is_held is find_held_objects,
                all_objs))
        else:
            objs = list(filter(lambda obj: obj.name == desired_obj.name and obj.location == location and
                isinstance(obj, Object) and obj.is_held is find_held_objects,
                all_objs))

        assert len(objs) == 1, "looking for {}, found {} at {}".format(desired_obj, ','.join(o.get_name() for o in objs), location)

        return objs[0]

    def get_gridsquare_at(self, location):
        gss = list(filter(lambda o: o.location == location and\
            isinstance(o, GridSquare), self.get_object_list()))

        assert len(gss) == 1, "{} gridsquares at {}: {}".format(len(gss), location, gss)
        return gss[0]

    def inbounds(self, location):
        """Correct locaiton to be in bounds of world object."""
        x, y = location
        return min(max(x, 0), self.width-1), min(max(y, 0), self.height-1)
