import recipe_planner.utils as recipe
from delegation_planner.delegator import Delegator
from delegation_planner.utils import SubtaskAllocDistribution
from navigation_planner.utils import get_subtask_obj, get_subtask_action_obj, get_single_actions
from utils.interact import interact
from utils.utils import agent_settings

from collections import defaultdict, namedtuple
from itertools import permutations, product, combinations
import scipy as sp
import numpy as np
import copy

SubtaskAllocation = namedtuple("SubtaskAllocation", "subtask subtask_agent_names")


class BayesianDelegator(Delegator):

    def __init__(self, agent_name, all_agent_names,
            model_type, planner, none_action_prob):
        """Initializing Bayesian Delegator for agent_name.

        Args:
            agent_name: Str of agent's name.
            all_agent_names: List of str agent names.
            model_type: Str of model type. Must be either "bd"=Bayesian Delegation,
                "fb"=Fixed Beliefs, "up"=Uniform Priors, "dc"=Divide & Conquer,
                "greedy"=Greedy.
            planner: Navigation Planner object, belonging to agent.
            none_action_prob: Float of probability for taking (0, 0) in a None subtask.
        """
        self.name = 'Bayesian Delegator'
        self.agent_name = agent_name
        self.all_agent_names = all_agent_names
        self.probs = None
        self.model_type = model_type
        self.priors = 'uniform' if model_type == 'up' else 'spatial'
        self.planner = planner
        self.none_action_prob = none_action_prob

    def should_reset_priors(self, obs, incomplete_subtasks):
        """Returns whether priors should be reset.

        Priors should be reset when 1) They haven't yet been set or
        2) If the possible subtask allocations to infer over have changed.

        Args:
            obs: Copy of the environment object. Current observation
                of environment.
            incomplete_subtasks: List of subtasks. Subtasks have not
                yet been completed according to agent.py.

        Return:
            Boolean of whether or not the subtask allocations have changed.
        """
        if self.probs is None:
            return True
        # Get currently available subtasks.
        self.incomplete_subtasks = incomplete_subtasks
        probs = self.get_subtask_alloc_probs()
        probs = self.prune_subtask_allocs(
                observation=obs, subtask_alloc_probs=probs)
        # Compare previously available subtasks with currently available subtasks.
        return not(len(self.probs.enumerate_subtask_allocs()) == len(probs.enumerate_subtask_allocs()))

    def get_subtask_alloc_probs(self):
        """Return the appropriate belief distribution (determined by model type) over
        subtask allocations (combinations of all_agent_names and incomplete_subtasks)."""
        if self.model_type == "greedy":
            probs = self.add_greedy_subtasks()
        elif self.model_type == "dc":
            probs = self.add_dc_subtasks()
        else:
            probs = self.add_subtasks()
        return probs

    def subtask_alloc_is_doable(self, env, subtask, subtask_agent_names):
        """Return whether subtask allocation (subtask x subtask_agent_names) is doable
        in the current environment state."""
        # Doing nothing is always possible.
        if subtask is None:
            return True
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, env.sim_agents))]
        start_obj, goal_obj = get_subtask_obj(subtask=subtask)
        subtask_action_obj = get_subtask_action_obj(subtask=subtask)
        A_locs, B_locs = env.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)
        distance = env.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs))
        # Subtask allocation is doable if it's reachable between agents and subtask objects.
        return distance < env.world.perimeter

    def get_lower_bound_for_subtask_alloc(self, obs, subtask, subtask_agent_names):
        """Return the value lower bound for a subtask allocation
        (subtask x subtask_agent_names)."""
        if subtask is None:
            return 0
        _ = self.planner.get_next_action(
                env=obs,
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners={})
        value = self.planner.v_l[(self.planner.cur_state.get_repr(), subtask)]
        return value

    def prune_subtask_allocs(self, observation, subtask_alloc_probs):
        """Removing subtask allocs from subtask_alloc_probs that are
        infeasible or where multiple agents are doing None together."""
        for subtask_alloc in subtask_alloc_probs.enumerate_subtask_allocs():
            for t in subtask_alloc:
                # Remove unreachable/undoable subtask subtask_allocations.
                if not self.subtask_alloc_is_doable(
                        env=observation,
                        subtask=t.subtask,
                        subtask_agent_names=t.subtask_agent_names):
                    subtask_alloc_probs.delete(subtask_alloc)
                    break
                # Remove joint Nones (cannot be collaborating on doing nothing).
                if t.subtask is None and len(t.subtask_agent_names) > 1:
                    subtask_alloc_probs.delete(subtask_alloc)
                    break

            # Remove all Nones (at least 1 agent must be doing something).
            if all([t.subtask is None for t in subtask_alloc]) and len(subtask_alloc) > 1:
                subtask_alloc_probs.delete(subtask_alloc)

        return subtask_alloc_probs

    def set_priors(self, obs, incomplete_subtasks, priors_type):
        """Setting the prior probabilities for subtask allocations."""
        print('{} setting priors'.format(self.agent_name))
        self.incomplete_subtasks = incomplete_subtasks

        probs = self.get_subtask_alloc_probs()
        probs = self.prune_subtask_allocs(
                observation=obs, subtask_alloc_probs=probs)
        probs.normalize()

        if priors_type == 'spatial':
            self.probs = self.get_spatial_priors(obs, probs)
        elif priors_type == 'uniform':
            # Do nothing because probs already initialized to be uniform.
            self.probs = probs

        self.ensure_at_least_one_subtask()
        self.probs.normalize()


    def get_spatial_priors(self, obs, some_probs):
        """Setting prior probabilities w.r.t spatial metrics."""
        # Weight inversely by distance.
        for subtask_alloc in some_probs.enumerate_subtask_allocs():
            total_weight = 0
            for t in subtask_alloc:
                if t.subtask is not None:
                    # Calculate prior with this agent's planner.
                    total_weight += 1.0 / float(self.get_lower_bound_for_subtask_alloc(
                        obs=copy.copy(obs),
                        subtask=t.subtask,
                        subtask_agent_names=t.subtask_agent_names))
            # Weight by number of nonzero subtasks.
            some_probs.update(
                    subtask_alloc=subtask_alloc,
                    factor=len(t)**2. * total_weight)
        return some_probs

    def get_other_agent_planners(self, obs, backup_subtask):
        """Use own beliefs to infer what other agents will do."""
        # A dictionary mapping agent name to a planner.
        # The planner is based on THIS agent's planner because agents are decentralized.
        planners = {}
        for other_agent_name in self.all_agent_names:
            # Skip over myself.
            if other_agent_name != self.agent_name:
                # Get most likely subtask and subtask agents for other agent
                # based on my beliefs.
                subtask, subtask_agent_names = self.select_subtask(
                        agent_name=other_agent_name)

                if subtask is None:
                    # Using cooperative backup_subtask for this agent's None subtask.
                    subtask = backup_subtask
                    subtask_agent_names = tuple(sorted([other_agent_name, self.agent_name]))

                # Assume your planner for other agents with the right settings.
                planner = copy.copy(self.planner)
                planner.set_settings(env=copy.copy(obs),
                                     subtask=subtask,
                                     subtask_agent_names=subtask_agent_names
                                     )
                planners[other_agent_name] = planner
        return planners

    def get_appropriate_state_and_other_agent_planners(self,
            obs_tm1, backup_subtask, no_level_1):
        """Return Level 1 planner if no_level_1 is False, otherwise
        return a Level 0 Planner."""
        # Get appropriate observation.
        if no_level_1:
            # Level 0 planning: Just use obs_tm1.
            state = obs_tm1
            # Assume other agents are fixed.
            other_planners = {}
        else:
            # Level 1 planning: Modify the state according to my beliefs.
            state, _ = self.planner._get_modified_state_with_other_agent_actions(state=obs_tm1)
            # Get other agent planners under my current beliefs.
            other_planners = self.get_other_agent_planners(
                    obs=obs_tm1, backup_subtask=backup_subtask)
        return state, other_planners

    def prob_nav_actions(self, obs_tm1, actions_tm1, subtask,
            subtask_agent_names, beta, no_level_1):
        """Return probabability that subtask_agents performed subtask, given
        previous observations (obs_tm1) and actions (actions_tm1).

        Args:
            obs_tm1: Copy of environment object. Represents environment at t-1.
            actions_tm1: Dictionary of agent actions. Maps agent str names to tuple actions.
            subtask: Subtask object to perform inference for.
            subtask_agent_names: Tuple of agent str names, of agents who perform subtask.
                subtask and subtask_agent_names make up subtask allocation.
            beta: Beta float value for softmax function.
            no_level_1: Bool, whether to turn off level-k planning.
        Returns:
            A float probability update of whether agents in subtask_agent_names are
            performing subtask.
        """
        print("[BayesianDelgation.prob_nav_actions] Calculating probs for subtask {} by {}".format(str(subtask), ' & '.join(subtask_agent_names)))
        assert len(subtask_agent_names) == 1 or len(subtask_agent_names) == 2

        # Perform inference over None subtasks.
        if subtask is None:
            assert len(subtask_agent_names) != 2, "Two agents are doing None."
            sim_agent = list(filter(lambda a: a.name == self.agent_name, obs_tm1.sim_agents))[0]
            # Get the number of possible actions at obs_tm1 available to agent.
            num_actions = len(get_single_actions(env=obs_tm1, agent=sim_agent)) -1
            action_prob = (1.0 - self.none_action_prob)/(num_actions)    # exclude (0, 0)
            diffs = [self.none_action_prob] + [action_prob] * num_actions
            softmax_diffs = sp.special.softmax(beta * np.asarray(diffs))
            # Probability agents did nothing for None subtask.
            if actions_tm1[subtask_agent_names[0]] == (0, 0):
                return softmax_diffs[0]
            # Probability agents moved for None subtask.
            else:
                return softmax_diffs[1]

        # Perform inference over all non-None subtasks.
        # Calculate Q_{subtask}(obs_tm1, action) for all actions.
        action = tuple([actions_tm1[a_name] for a_name in subtask_agent_names])
        if len(subtask_agent_names) == 1:
            action = action[0]
        state, other_planners = self.get_appropriate_state_and_other_agent_planners(
                obs_tm1=obs_tm1, backup_subtask=subtask, no_level_1=no_level_1)
        self.planner.set_settings(env=obs_tm1, subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners=other_planners)
        old_q = self.planner.Q(state=state, action=action,
                value_f=self.planner.v_l)

        # Collect actions the agents could have taken in obs_tm1.
        valid_nav_actions = self.planner.get_actions(state_repr=obs_tm1.get_repr())

        # Check action taken is in the list of actions available to agents in obs_tm1.
        assert action in valid_nav_actions, "valid_nav_actions: {}\nlocs: {}\naction: {}".format(
                valid_nav_actions, list(filter(lambda a: a.location, state.sim_agents)), action)

        # If subtask allocation is joint, then find joint actions that match what the other
        # agent's action_tm1.
        if len(subtask_agent_names) == 2 and self.agent_name in subtask_agent_names:
            other_index = 1 - subtask_agent_names.index(self.agent_name)
            valid_nav_actions = list(filter(lambda x: x[other_index] == action[other_index], valid_nav_actions))

        # Calculating the softmax Q_{subtask} for each action.
        qdiffs = [old_q - self.planner.Q(state=state, action=nav_action, value_f=self.planner.v_l)
                for nav_action in valid_nav_actions]
        softmax_diffs = sp.special.softmax(beta * np.asarray(qdiffs))
        # Taking the softmax of the action actually taken.
        return softmax_diffs[valid_nav_actions.index(action)]

    def get_other_subtask_allocations(self, remaining_agents, remaining_subtasks, base_subtask_alloc):
        """Return a list of subtask allocations to be added onto `subtask_allocs`.

        Each combination should be built off of the `base_subtask_alloc`.
        Add subtasks for all other agents and all other recipe subtasks NOT in
        the ignore set.

        e.g. base_subtask_combo=[
            SubtaskAllocation(subtask=(Chop(T)),
            subtask_agent_names(agent-1, agent-2))]
        To be added on: [
            SubtaskAllocation(subtask=(Chop(L)),
            subtask_agent_names(agent-3,))]
        Note the different subtask and the different agent.
        """
        other_subtask_allocs = []
        if not remaining_agents:
            return [base_subtask_alloc]

        # This case is hit if we have more agents than subtasks.
        if not remaining_subtasks:
            for agent in remaining_agents:
                new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=None, subtask_agent_names=tuple(agent))]
                other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs

        # Otherwise assign remaining agents to remaining subtasks.
        # If only 1 agent left, assign to all remaining subtasks.
        if len(remaining_agents) == 1:
            for t in remaining_subtasks:
                new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(remaining_agents))]
                other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs
        # If >1 agent remaining, create cooperative and divide & conquer
        # subtask allocations.
        else:
            # Cooperative subtasks (same subtask assigned to remaining agents).
            for t in remaining_subtasks:
                new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(remaining_agents))]
                other_subtask_allocs.append(new_subtask_alloc)
            # Divide and Conquer subtasks (different subtask assigned to remaining agents).
            if len(remaining_subtasks) > 1:
                for ts in permutations(remaining_subtasks, 2):
                    new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=ts[0], subtask_agent_names=(remaining_agents[0], )),
                                                   SubtaskAllocation(subtask=ts[1], subtask_agent_names=(remaining_agents[1], )),]
                    other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs

    def add_subtasks(self):
        """Return the entire distribution of subtask allocations."""
        subtask_allocs = []

        subtasks = self.incomplete_subtasks
        # Just one agent: Assign itself to all subtasks.
        if len(self.all_agent_names) == 1:
            for t in subtasks:
                subtask_alloc = [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(self.all_agent_names))]

                subtask_allocs.append(subtask_alloc)
        else:
            for first_agents in combinations(self.all_agent_names, 2):
                # Temporarily add Nones, to allow agents to be allocated no subtask.
                # Later, we filter out allocations where all agents are assigned to None.
                subtasks_temp = subtasks + [None for _ in range(len(self.all_agent_names) - 1)]
                # Cooperative subtasks (same subtask assigned to agents).
                for t in subtasks_temp:
                    subtask_alloc = [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(first_agents))]
                    remaining_agents = sorted(list(set(self.all_agent_names) - set(first_agents)))
                    remaining_subtasks = list(set(subtasks_temp) - set([t]))
                    subtask_allocs += self.get_other_subtask_allocations(
                            remaining_agents=remaining_agents,
                            remaining_subtasks=remaining_subtasks,
                            base_subtask_alloc=subtask_alloc)
                # Divide and Conquer subtasks (different subtask assigned to remaining agents).
                if len(subtasks_temp) > 1:
                    for ts in permutations(subtasks_temp, 2):
                        subtask_alloc = [
                                SubtaskAllocation(subtask=ts[0], subtask_agent_names=(first_agents[0],)),
                                SubtaskAllocation(subtask=ts[1], subtask_agent_names=(first_agents[1],)),]
                        remaining_agents = sorted(list(set(self.all_agent_names) - set(first_agents)))
                        remaining_subtasks = list(set(subtasks_temp) - set(ts))
                        subtask_allocs += self.get_other_subtask_allocations(
                                remaining_agents=remaining_agents,
                                remaining_subtasks=remaining_subtasks,
                                base_subtask_alloc=subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)

    def add_greedy_subtasks(self):
        """Return the entire distribution of greedy subtask allocations.
        i.e. subtasks performed only by agent with self.agent_name."""
        subtask_allocs = []

        subtasks = self.incomplete_subtasks
        # At least 1 agent must be doing something.
        if None not in subtasks:
            subtasks += [None]

        # Assign this agent to all subtasks. No joint subtasks because this function
        # only considers greedy subtask allocations.
        for subtask in subtasks:
            subtask_alloc = [SubtaskAllocation(subtask=subtask, subtask_agent_names=(self.agent_name,))]
            subtask_allocs.append(subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)

    def add_dc_subtasks(self):
        """Return the entire distribution of divide & conquer subtask allocations.
        i.e. no subtask is shared between two agents.

        If there are no subtasks, just make an empty distribution and return."""
        subtask_allocs = []

        subtasks = self.incomplete_subtasks + [None for _ in range(len(self.all_agent_names) - 1)]
        for p in permutations(subtasks, len(self.all_agent_names)):
            subtask_alloc = [SubtaskAllocation(subtask=p[i], subtask_agent_names=(self.all_agent_names[i],)) for i in range(len(self.all_agent_names))]
            subtask_allocs.append(subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)

    def select_subtask(self, agent_name):
        """Return subtask and subtask_agent_names for agent with agent_name
        with max. probability."""
        max_subtask_alloc = self.probs.get_max()
        if max_subtask_alloc is not None:
            for t in max_subtask_alloc:
                if agent_name in t.subtask_agent_names:
                    return t.subtask, t.subtask_agent_names
        return None, agent_name

    def ensure_at_least_one_subtask(self):
        # Make sure each agent has None task by itself.
        if (self.model_type == "greedy" or self.model_type == "dc"):
            if not self.probs.probs:
                subtask_allocs = [[SubtaskAllocation(subtask=None, subtask_agent_names=(self.agent_name,))]]
                self.probs = SubtaskAllocDistribution(subtask_allocs)

    def bayes_update(self, obs_tm1, actions_tm1, beta):
        """Apply Bayesian update based on previous observation (obs_tms1)
        and most recent actions taken (actions_tm1). Beta is used to determine
        how rational agents act."""
        # First, remove unreachable/undoable subtask agent subtask_allocs.
        for subtask_alloc in self.probs.enumerate_subtask_allocs():
            for t in subtask_alloc:
                if not self.subtask_alloc_is_doable(
                        env=obs_tm1,
                        subtask=t.subtask,
                        subtask_agent_names=t.subtask_agent_names):
                    self.probs.delete(subtask_alloc)
                    break

        self.ensure_at_least_one_subtask()

        if self.model_type  == "fb":
            return

        for subtask_alloc in self.probs.enumerate_subtask_allocs():
            update = 0.0
            for t in subtask_alloc:
                if self.model_type == "greedy":
                    # Only calculate updates for yourself.
                    if self.agent_name in t.subtask_agent_names:
                        update += self.prob_nav_actions(
                                obs_tm1=copy.copy(obs_tm1),
                                actions_tm1=actions_tm1,
                                subtask=t.subtask,
                                subtask_agent_names=t.subtask_agent_names,
                                beta=beta,
                                no_level_1=False)
                else:
                    p = self.prob_nav_actions(
                            obs_tm1=copy.copy(obs_tm1),
                            actions_tm1=actions_tm1,
                            subtask=t.subtask,
                            subtask_agent_names=t.subtask_agent_names,
                            beta=beta,
                            no_level_1=False)
                    update += len(t.subtask_agent_names) * p

            self.probs.update(
                    subtask_alloc=subtask_alloc,
                    factor=update)
            print("UPDATING: subtask_alloc {} by {}".format(subtask_alloc, update))
        self.probs.normalize()
