from utils.core import *
import recipe_planner.utils as recipe


class Recipe:
    def __init__(self, name):
        self.name = name
        self.contents = []
        self.actions = set()
        self.actions.add(recipe.Get('Plate'))

    def __str__(self):
        return self.name

    def add_ingredient(self, item):
        self.contents.append(item)

        # always starts with FRESH
        self.actions.add(recipe.Get(item.name))

        if item.state_seq == FoodSequence.FRESH_CHOPPED:
            self.actions.add(recipe.Chop(item.name))
            self.actions.add(recipe.Merge(item.name, 'Plate',\
                [item.state_seq[-1](item.name), recipe.Fresh('Plate')], None))

    def add_goal(self):
        self.contents = sorted(self.contents, key = lambda x: x.name)   # list of Food objects
        self.contents_names = [c.name for c in self.contents]   # list of strings
        self.full_name = '-'.join(sorted(self.contents_names))   # string
        self.full_plate_name = '-'.join(sorted(self.contents_names + ['Plate']))   # string
        self.goal = recipe.Delivered(self.full_plate_name)
        self.actions.add(recipe.Deliver(self.full_plate_name))

    def add_merge_actions(self):
        # should be general enough for any kind of salad / raw plated veggies

        # alphabetical, joined by dashes ex. Ingredient1-Ingredient2-Plate
        #self.full_name = '-'.join(sorted(self.contents + ['Plate']))

        # for any plural number of ingredients
        for i in range(2, len(self.contents)+1):
            # for any combo of i ingredients to be merged
            for combo in combinations(self.contents_names, i):
                # can merge all with plate
                self.actions.add(recipe.Merge('-'.join(sorted(combo)), 'Plate',\
                    [recipe.Merged('-'.join(sorted(combo))), recipe.Fresh('Plate')], None))

                # for any one item to be added to the i-1 rest
                for item in combo:
                    rem = list(combo).copy()
                    rem.remove(item)
                    rem_str = '-'.join(sorted(rem))
                    plate_str = '-'.join(sorted([item, 'Plate']))
                    rem_plate_str = '-'.join(sorted(rem + ['Plate']))

                    # can merge item with remaining
                    if len(rem) == 1:
                        self.actions.add(recipe.Merge(item, rem_str,\
                            [recipe.Chopped(item), recipe.Chopped(rem_str)], None))
                        self.actions.add(recipe.Merge(rem_str, plate_str))
                        self.actions.add(recipe.Merge(item, rem_plate_str))
                    else:
                        self.actions.add(recipe.Merge(item, rem_str))
                        self.actions.add(recipe.Merge(plate_str, rem_str,\
                            [recipe.Merged(plate_str), recipe.Merged(rem_str)], None))
                        self.actions.add(recipe.Merge(item, rem_plate_str))

class SimpleTomato(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Tomato')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_goal()
        self.add_merge_actions()

class SimpleLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Lettuce')
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()

class Salad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Salad')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()

class OnionSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionSalad')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_ingredient(Onion(state_index=-1))
        self.add_goal()
        self.add_merge_actions()


