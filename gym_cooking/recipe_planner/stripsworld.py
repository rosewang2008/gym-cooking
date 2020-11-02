import recipe_planner.utils as recipe

# core modules
from utils.core import Object

# helpers
import networkx as nx
import copy


class STRIPSWorld:
    def __init__(self, world, recipes):
        self.initial = recipe.STRIPSState()
        self.recipes = recipes

        # set initial state
        self.initial.add_predicate(recipe.NoPredicate())
        for obj in world.get_object_list():
            if isinstance(obj, Object):
                for obj_name in ['Plate', 'Tomato', 'Lettuce', 'Onion']:
                    if obj.contains(obj_name):
                        self.initial.add_predicate(recipe.Fresh(obj_name))

    def generate_graph(self, recipe, max_path_length):
        all_actions = recipe.actions   # set
        goal_state = None

        new_preds = set()
        graph = nx.DiGraph()
        graph.add_node(self.initial, obj=self.initial)
        frontier = set([self.initial])
        next_frontier = set()
        for i in range(max_path_length):
            # print('CHECKING FRONTIER #:', i)
            for state in frontier:
                # for each action, check whether from this state
                for a in all_actions:
                    if a.is_valid_in(state):
                        next_state = a.get_next_from(state)
                        for p in next_state.predicates:
                            new_preds.add(str(p))
                        graph.add_node(next_state, obj=next_state)
                        graph.add_edge(state, next_state, obj=a)

                        # as soon as goal is found, break and return                       
                        if self.check_goal(recipe, next_state) and goal_state is None:
                            goal_state = next_state
                            return graph, goal_state
                        
                        next_frontier.add(next_state)

            frontier = next_frontier.copy()
        
        if goal_state is None:
            print('goal state could not be found, try increasing --max-num-subtasks')
            import sys; sys.exit(0)
        
        return graph, goal_state


    def get_subtasks(self, max_path_length=10, draw_graph=False):
        action_paths = []

        for recipe in self.recipes:
            graph, goal_state = self.generate_graph(recipe, max_path_length)

            if draw_graph:   # not recommended for path length > 4
                nx.draw(graph, with_labels=True)
                plt.show()
            
            all_state_paths = nx.all_shortest_paths(graph, self.initial, goal_state)
            union_action_path = set()
            for state_path in all_state_paths:
                action_path = [graph[state_path[i]][state_path[i+1]]['obj'] for i in range(len(state_path)-1)]
                union_action_path = union_action_path | set(action_path)
            # print('all tasks for recipe {}: {}\n'.format(recipe, ', '.join([str(a) for a in union_action_path])))
            action_paths.append(union_action_path)

        return action_paths
        

    def check_goal(self, recipe, state):
        # check if this state satisfies completion of this recipe
        return state.contains(recipe.goal)




