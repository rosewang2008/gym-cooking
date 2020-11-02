# Recipe planning
from recipe_planner.utils import *

# Navigation planning
import navigation_planner.utils as nav_utils
from navigation_planner.utils import MinPriorityQueue as mpq

# Other core modules
from utils.world import World
from utils.interact import interact
from utils.core import *

from collections import defaultdict
import numpy as np
import scipy as sp
import random
from itertools import product
import copy
import time
from functools import lru_cache
from enum import Enum

class PlannerLevel(Enum):
    LEVEL1 = 1
    LEVEL0 = 0

def argmin(vector):
    e_x = np.array(vector) == min(vector)

    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]

def argmax(vector):
    e_x = np.array(vector) == max(vector)
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]

class E2E_BRTDP:
    """Bounded Real Time Dynamic Programming (BRTDP) algorithm.

    For more details on this algorithm, please refer to the original
    paper: http://www.cs.cmu.edu/~ggordon/mcmahan-likhachev-gordon.brtdp.pdf
    """

    def __init__(self, alpha, tau, cap, main_cap):
        """
        Initializes BRTDP algorithm with its hyper-parameters.
        Rf. BRTDP paper for how these hyper-parameters are used in their
        algorithm.

        http://www.cs.cmu.edu/~ggordon/mcmahan-likhachev-gordon.brtdp.pdf

        Args:
            alpha: BRTDP convergence criteria.
            tau: BRTDP normalization constant.
            cap: BRTDP cap on sample trial rollouts.
            main_cap: BRTDP main cap on its main loop.
        """
        self.alpha = alpha
        self.tau = tau
        self.cap = cap
        self.main_cap = main_cap

        self.v_l = {}
        self.v_u = {}
        self.repr_to_env_dict = dict()
        self.start = None
        self.pq = mpq()
        self.actions = World.NAV_ACTIONS
        self.is_joint = False
        self.planner_level = PlannerLevel.LEVEL0
        self.cur_object_count = 0
        self.is_subtask_complete = lambda h: False
        self.removed_object = None
        self.goal_obj = None

        # Setting up costs for value function.
        self.time_cost = 1.0
        self.action_cost = 0.1

    def __copy__(self):
        copy_ = E2E_BRTDP(
                    alpha=self.alpha, tau=self.tau,
                    cap=self.cap, main_cap=self.main_cap)
        copy_.__dict__ = self.__dict__.copy()
        return copy_

    @lru_cache(maxsize=10000)
    def T(self, state_repr, action):
        """Return next states when taking action from state."""
        state = self.repr_to_env_dict[state_repr]
        subtask_agents = self.get_subtask_agents(env_state=state)

        # Single agent
        if not self.is_joint:
            agent = subtask_agents[0]
            sim_state = copy.copy(state)
            sim_agent = list(filter(lambda a: a.name == agent.name, sim_state.sim_agents))[0]
            sim_agent.action = action
            interact(agent=sim_agent,
                     world=sim_state.world)

        # Joint
        else:
            agent_1, agent_2 = subtask_agents
            sim_state = copy.copy(state)
            sim_agent_1 = list(filter(lambda a: a.name == agent_1.name, sim_state.sim_agents))[0]
            sim_agent_2 = list(filter(lambda a: a.name == agent_2.name, sim_state.sim_agents))[0]
            sim_agent_1.action, sim_agent_2.action = action
            interact(agent=sim_agent_1, world=sim_state.world)
            interact(agent=sim_agent_2, world=sim_state.world)
            assert sim_agent_1.location != sim_agent_2.location, 'action {} led to state {}'.format(action, sim_state.get_repr())

        # Track this state in value function and repr dict
        # if it's a new state.
        self.repr_init(env_state=sim_state)
        self.value_init(env_state=sim_state)
        return sim_state

    def get_actions(self, state_repr):
        """Returns list of possible actions from current state."""
        if self.subtask is None:
            return [(0, 0)]
        # Convert repr into an environment object.
        state = self.repr_to_env_dict[state_repr]

        subtask_agents = self.get_subtask_agents(env_state=state)
        output_actions = []

        # Return single-agent actions.
        if not self.is_joint:
            agent = subtask_agents[0]
            output_actions = nav_utils.get_single_actions(env=state, agent=agent)
        # Return joint-agent actions.
        else:
            agent_1, agent_2 = subtask_agents
            valid_actions = list(product(
                nav_utils.get_single_actions(env=state, agent=agent_1),
                nav_utils.get_single_actions(env=state, agent=agent_2)))
            # Only consider action to be valid if agents do not collide.
            for va in valid_actions:
                agent1, agent2 = va
                execute = state.is_collision(
                        agent1_loc=agent_1.location,
                        agent2_loc=agent_2.location,
                        agent1_action=agent1,
                        agent2_action=agent2)
                if all(execute):
                    output_actions.append(va)
        return output_actions


    def runSampleTrial(self):
        """runSampleTrial from BRTDP paper."""
        start_time = time.time()
        x = self.start
        traj = nav_utils.Stack()

        # Terminating if this takes too long e.g. path is infeasible.
        counter = 0
        start_repr = self.start.get_repr()
        diff = self.v_u[(start_repr, self.subtask)] - self.v_l[(start_repr, self.subtask)]
        print("DIFF AT START: {}".format(diff))

        while True:
            counter += 1
            if counter > self.cap:
                break
            traj.push(x)

            # Get repr of current environment state.
            x_repr = x.get_repr()

            # Get the planner state. If Planner Level is 1, then
            # modified_state will include the most likely actions of the
            # other agents. Otherwise, the modified_state will be the same
            # as state `x`.
            modified_state, other_agent_actions = self._get_modified_state_with_other_agent_actions(x)
            modified_state_repr = modified_state.get_repr()

            # Get available actions from this state.
            actions = self.get_actions(state_repr=modified_state_repr)

            # We pick actions based on expected state.
            new_upper = min([
                self.Q(state=modified_state, action=a, value_f=self.v_u)
                for a in actions])
            self.v_u[(modified_state_repr, self.subtask)] = new_upper

            action_index = argmin([
                self.Q(state=modified_state, action=a, value_f=self.v_l)
                for a in actions])
            a = actions[action_index]

            new_lower = self.Q(state=modified_state, action=a, value_f=self.v_l)
            self.v_l[(modified_state_repr, self.subtask)] = new_lower

            b = self.get_expected_diff(modified_state, a)
            B = sum(b.values())
            diff = (self.v_u[(start_repr, self.subtask)] - self.v_l[(start_repr, self.subtask)])/self.tau
            if (B <= diff):
                break

            x = self.repr_to_env_dict[list(b.keys())[0]]

            # Track this new state in repr dict and value function
            # if it's new.
            self.repr_init(env_state=x)
            self.value_init(env_state=x)

        print("RUN SAMPLE EXPLORED {} STATES, took {}".format(len(traj), time.time()-start_time))
        while not(traj.empty()):
            x = traj.pop()
            x_repr = x.get_repr()
            actions = self.get_actions(state_repr=x_repr)
            self.v_u[(x_repr, self.subtask)] = min([
                self.Q(state=x, action=a, value_f=self.v_u) for a in actions])
            self.v_l[(x_repr, self.subtask)] = min([
                self.Q(state=x, action=a, value_f=self.v_l) for a in actions])

    def main(self):
        """Main loop function for BRTDP."""
        main_counter = 0
        start_repr = self.start.get_repr()

        upper = self.v_u[(start_repr, self.subtask)]
        lower = self.v_l[(start_repr, self.subtask)]
        diff = upper - lower

        # Run until convergence or until you max out on iteration
        while (diff > self.alpha) and (main_counter < self.main_cap):
            print('\nstarting main loop #', main_counter)
            new_upper = self.v_u[(start_repr, self.subtask)]
            new_lower = self.v_l[(start_repr, self.subtask)]
            new_diff = new_upper - new_lower
            if new_diff > diff + 0.01:
                self.start.update_display()
                self.start.display()
                self.start.print_agents()
                print('old: upper {}, lower {}'.format(upper, lower))
                print('new: upper {}, lower {}'.format(new_upper, new_lower))
            diff = new_diff
            upper = new_upper
            lower = new_lower
            main_counter +=1
            print('diff = {}, self.alpha = {}'.format(diff, self.alpha))
            self.runSampleTrial()

    def _configure_planner_level(self, env, subtask_agent_names, other_agent_planners):
        """Configure the planner s.t. it best responds to other agents as needed.

        If other_agent_planners is an emtpy dict, then this planner should
        be a level-0 planner and remove all irrelevant agents in env.

        Otherwise, it should keep all agents and maintain their planners
        which have already been configured to the subtasks we believe them to
        have."""
        # Level 1 planner
        if other_agent_planners:
            self.planner_level = PlannerLevel.LEVEL1
            self.other_agent_planners = other_agent_planners
        # Level 0 Planner.
        else:
            self.planner_level = PlannerLevel.LEVEL0
            self.other_agent_planners = {}
            # Replace other agents with counters (frozen agents during planning).
            rm_agents = []
            for agent in env.sim_agents:
                if agent.name not in subtask_agent_names:
                    rm_agents.append(agent)
            for agent in rm_agents:
                env.sim_agents.remove(agent)
                if agent.holding is not None:
                    self.removed_object = agent.holding
                    env.world.remove(agent.holding)

                # Remove Floor and replace with Counter. This is needed when
                # checking whether object @ location is collidable.
                env.world.remove(Floor(agent.location))
                env.world.insert(AgentCounter(agent.location))

    def _configure_subtask_information(self, subtask, subtask_agent_names):
        """Tracking information about subtask allocation."""
        # Subtask allocation
        self.subtask = subtask
        self.subtask_agent_names = subtask_agent_names

        # Relevant objects for subtask allocation.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
        self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)

    def _define_goal_state(self, env, subtask):
        """Defining a goal state (termination condition on state) for subtask."""

        if subtask is None:
            self.is_goal_state = lambda h: True

        # Termination condition is when desired object is at a Deliver location.
        elif isinstance(subtask, Deliver):
            self.cur_obj_count = len(
                    list(filter(lambda o: o in set(env.world.get_all_object_locs(
                            self.subtask_action_obj)),
                    env.world.get_object_locs(self.goal_obj, is_held=False))))
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_goal_state = lambda h: self.has_more_obj(
                    len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                    self.repr_to_env_dict[h].world.get_object_locs(self.goal_obj, is_held=False)))))

            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                        w.get_object_locs(self.goal_obj, is_held=False)))) + 1)
            else:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                        w.get_object_locs(obj=self.goal_obj, is_held=False)))))
        else:
            # Get current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_goal_state = lambda h: self.has_more_obj(
                    len(self.repr_to_env_dict[h].world.get_all_object_locs(self.goal_obj)))
            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)) + 1)
            else:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)))

    def _configure_planner_space(self, subtask_agent_names):
        """Configure planner to either plan in joint space or single-agent space."""
        assert len(subtask_agent_names) <= 2, "Cannot have more than 2 agents! Hm... {}".format(subtask_agents)

        self.is_joint = len(subtask_agent_names) == 2

    def set_settings(self, env, subtask, subtask_agent_names, other_agent_planners={}):
        """Configure planner."""
        # Configuring the planner level.
        self._configure_planner_level(
                env=env,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners=other_agent_planners)

        # Configuring subtask related information.
        self._configure_subtask_information(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names)

        # Defining what the goal is for this planner.
        self._define_goal_state(
                env=env,
                subtask=subtask)

        # Defining the space of the planner (joint or single).
        self._configure_planner_space(subtask_agent_names=subtask_agent_names)

        # Make sure termination counter has been reset.
        self.counter = 0
        self.num_explorations = 0
        self.stop = False
        self.num_explorations = 0

        # Set start state.
        self.start = copy.copy(env)
        self.repr_init(env_state=env)
        self.value_init(env_state=env)

    def get_subtask_agents(self, env_state):
        """Return subtask agent for this planner given state."""
        subtask_agents = list(filter(lambda a: a.name in self.subtask_agent_names, env_state.sim_agents))

        assert list(map(lambda a: a.name, subtask_agents)) == list(self.subtask_agent_names), "subtask agent names are not in order: {} != {}".format(list(map(lambda a: a.name, subtask_agents)), self.subtask_agent_names)

        return subtask_agents

    def repr_init(self, env_state):
        """Initialize repr for environment state."""
        es_repr = env_state.get_repr()
        if es_repr not in self.repr_to_env_dict:
            self.repr_to_env_dict[es_repr] = copy.copy(env_state)
        return es_repr

    def value_init(self, env_state):
        """Initialize value for environment state."""
        # Skip if already initialized.
        es_repr = env_state.get_repr()
        if ((es_repr, self.subtask) in self.v_l and
            (es_repr, self.subtask) in self.v_u):
            return

        # Goal state has value 0.
        if self.is_goal_state(es_repr):
            self.v_l[(es_repr, self.subtask)] = 0.0
            self.v_u[(es_repr, self.subtask)] = 0.0
            return

        # Determine lower bound on this environment state.
        lower = env_state.get_lower_bound_for_subtask_given_objs(
                subtask=self.subtask,
                subtask_agent_names=self.subtask_agent_names,
                start_obj=self.start_obj,
                goal_obj=self.goal_obj,
                subtask_action_obj=self.subtask_action_obj)

        subtask_agents = self.get_subtask_agents(env_state=env_state)
        lower = lower * (self.time_cost + self.action_cost)

        # By BRTDP assumption, this should never be negative.
        assert lower > 0, "lower: {}, {}, {}".format(lower, env_state.display(), env_state.print_agents())

        self.v_l[(es_repr, self.subtask)] = lower - 1.09
        self.v_u[(es_repr, self.subtask)] = lower * 5 * (self.time_cost + self.action_cost)


    def Q(self, state, action, value_f):
        """Get Q value using value_f of (state, action)."""
        # Q(s,a) = c(x,a) + \sum_{y \in S} P(x, a, y) * v(y)
        cost = self.cost(state, action)

        # Initialize state if it's new.
        s_repr = self.repr_init(env_state=state)
        self.value_init(env_state=state)

        # Get next state.
        next_state = self.T(state_repr=s_repr, action=action)

        # Initialize new state if it's new.
        ns_repr = self.repr_init(env_state=next_state)
        self.value_init(env_state=next_state)

        expected_value = 1.0 * value_f[(ns_repr, self.subtask)]
        return float(cost + expected_value)

    def V(self, state, _type):
        """Get V*(x) = min_{a \in A} Q_{v*}(x, a)."""

        # Initialize state if it's new.
        s_repr = self.repr_init(env_state=state)

        # Check if this is the desired goal state.
        if self.is_goal_state(s_repr):
            return 0

        # Use lower bound on value function.
        if _type == "lower":
            return min([
                self.Q(state=state, action=action, value_f=self.v_l)
                for action in self.get_actions(state_repr=s_repr)])
        # Use upper bound on value function.
        elif _type == "upper":
            return min([
                self.Q(state=state, action=action, value_f=self.v_u)
                for action in self.get_actions(state_repr=s_repr)])
        else:
            raise ValueError("Don't recognize the value state function type: {}".format(_type))

    def cost(self, state, action):
        """Return cost of taking action in this state."""
        cost = self.time_cost
        if isinstance(action[0], int):
            action = tuple([action])
        for a in action:
            if a != (0, 0):
                cost += self.action_cost
        return cost

    def get_expected_diff(self, start_state, action):
        # Get next state.
        s_ = self.T(state_repr=start_state.get_repr(), action=action)

        # Initialize state if it's new.
        s_repr = self.repr_init(env_state=s_)
        self.value_init(env_state=s_)

        # Get expected diff.
        b = {s_repr: 1.0 * (self.v_u[(s_repr, self.subtask)] - self.v_l[(s_repr, self.subtask)])}
        return b

    def _get_modified_state_with_other_agent_actions(self, state):
        """Do nothing if the planner level is level 0.

        Otherwise, using self.other_agent_planners, anticipate what other agents will do
        and modify the state appropriately.

        Returns the modified state and the actions of other agents that triggered
        the change.
        """
        modified_state = copy.copy(state)
        other_agent_actions = {}

        # Do nothing if the planner level is 0.
        if self.planner_level == PlannerLevel.LEVEL0:
            return modified_state, other_agent_actions

        # Otherwise, modify the state because Level 1 planners
        # consider the actions of other agents.
        for other_agent_name, other_planner in self.other_agent_planners.items():
            # Keep their recipe subtask & subtask agent fixed, but change
            # their planner state to `state`.
            # These new planners should be level 0 planners.
            other_planner.set_settings(env=copy.copy(state),
                                       subtask=other_planner.subtask,
                                       subtask_agent_names=other_planner.subtask_agent_names)

            assert other_planner.planner_level == PlannerLevel.LEVEL0

            # Figure out what their most likely action is.
            possible_actions = other_planner.get_actions(state_repr=other_planner.start.get_repr())
            greedy_action = possible_actions[
                    argmin([other_planner.Q(state=other_planner.start,
                                            action=action,
                                            value_f=other_planner.v_l)
                    for action in possible_actions])]

            if other_planner.is_joint:
                greedy_action = greedy_action[other_planner.subtask_agent_names.index(other_agent_name)]

            # Keep track of their actions.
            other_agent_actions[other_agent_name] = greedy_action
            other_agent = list(filter(lambda a: a.name == other_agent_name,
                                      modified_state.sim_agents))[0]
            other_agent.action = greedy_action

        # Initialize state if it's new.
        self.repr_init(env_state=modified_state)
        self.value_init(env_state=modified_state)
        return modified_state, other_agent_actions

    def get_next_action(self, env, subtask, subtask_agent_names, other_agent_planners):
        """Return next action."""
        print("-------------[e2e]-----------")
        self.removed_object = None
        start_time = time.time()

        # Configure planner settings.
        self.set_settings(
                env=env, subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners=other_agent_planners)

        # Modify the state with other_agent_planners (Level 1 Planning).
        cur_state, other_agent_actions = self._get_modified_state_with_other_agent_actions(state=self.start)

        # BRTDP main loop.
        actions = self.get_actions(state_repr=cur_state.get_repr())
        action_index = argmin([
            self.Q(state=cur_state, action=a, value_f=self.v_l)
            for a in actions])
        a = actions[action_index]
        B = sum(self.get_expected_diff(cur_state, a).values())
        diff = (self.v_u[(cur_state.get_repr(), self.subtask)] - self.v_l[(cur_state.get_repr(), self.subtask)])/self.tau
        self.cur_state = cur_state
        if (B > diff):
            print('exploring, B: {}, diff: {}'.format(B, diff))
            self.main()

        # Determine best action after BRTDP.
        if self.is_goal_state(cur_state.get_repr()):
            print('already at goal state, self.cur_obj_count:', self.cur_obj_count)
            return None
        else:
            actions = self.get_actions(state_repr=cur_state.get_repr())
            qvals = [self.Q(state=cur_state, action=a, value_f=self.v_l)
                    for a in actions]
            print([x for x in zip(actions, qvals)])
            print('upper is', self.v_u[(cur_state.get_repr(), self.subtask)])
            print('lower is', self.v_l[(cur_state.get_repr(), self.subtask)])

            action_index = argmin(np.array(qvals))
            a = actions[action_index]

            print('chose action:', a)
            print('cost:', self.cost(cur_state, a))
            return a
