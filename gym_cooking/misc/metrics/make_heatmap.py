import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import pandas as pd
import itertools

import numpy as np
import pickle
import sys
from collections import defaultdict
import os
sys.path.append("../..")
import recipe_planner
import seaborn as sns

# NOTE: THIS SCRIPT IS ONLY WRITTEN FOR 2 AGENTS

split_by_env = False
recipes = [
    'tomato',
        "tl",
    'salad'
        ]
models = [
    "bd",
    "up",
    "fb",
    "dc",
    "greedy",
]
model_key = {
    "bd": 'BD (ours)',
    "up": 'UP',
    "fb": "FB",
    "dc": "D&C",
    "greedy": "Greedy",
}
maps = [
        "full-divider",
        "open-divider",
        "partial-divider"
        ]
seeds = range(1,16)
num_agents = 2

path_pickles = os.path.join(os.getcwd(), 'pickles')
path_save = os.path.join(os.getcwd(), 'heatmaps')
if not os.path.exists(path_save):
    os.mkdir(path_save)

am = {'agent-1': 'Agent-1', 'agent-2': 'Agent-2',}
tm = {
    'Chop(Lettuce)': 'Merge(Lettuce, Cutting Board)',
    'Chop(Tomato)': 'Merge(Tomato, Cutting Board)',
    'Merge(Tomato, Plate)': 'Merge(Tomato, Plate[])',
    'Merge(Lettuce, Plate)': 'Merge(Lettuce, Plate[])',
    'Deliver(Plate-Tomato)': 'Merge(Plate[Tomato], Delivery[])',
    'Deliver(Lettuce-Plate)': 'Merge(Plate[Lettuce], Delivery[])',
    'Deliver(Lettuce-Plate-Tomato)': 'Merge(Plate[Lettuce, Tomato], Delivery[])',
    'None': 'None',
}
objects = [
    ("Fresh Tomato", "Chopped Tomato", True),
    ("Chopped Tomato", "cleanChopped Plate-Tomato", False),
    ("Fresh Lettuce", "Chopped Lettuce", True),
    ("Chopped Lettuce", "cleanChopped Lettuce-Plate", False),
]

def get_time_steps(data, recipe):
    if data['was_successful']:
        return len(data['num_completed_tasks'])
    else: return 100

df = list()
run_id = 0
for recipe, model1, model2, map_, seed in itertools.product(recipes, models, models, maps, seeds):
    fname = '{}_{}_agents{}_seed{}_model1-{}_model2-{}.pkl'.format(map_, recipe, num_agents, seed, model1, model2)

    if os.path.exists(os.path.join(path_pickles, fname)):
        try:
            data = pickle.load(open(os.path.join(path_pickles, fname), "rb"))
        except:
            continue
    else:
        print('no file:', fname)
        continue
    info = {
        'run_id': run_id,
        'model1':model1,
        'model2':model2,
        'recipe': recipe,
        'map': map_,
        'seed': seed,
        'dummy': 0,
    }
    time_steps = get_time_steps(data, recipe)
    df.append(dict(time_steps = time_steps, **info))


df = pd.DataFrame(df)


# Plotting aggregate
new_df = list()
for model in models:
    insert_data = np.array(df.loc[(df['model1']==model) |
                  (df['model2']==model)
                  , :]['time_steps'])
    for d in insert_data:
        new_df.append(dict({
            'time_steps':d,
            'model': model,
            },))
new_df = pd.DataFrame(new_df)
ax = sns.barplot(x="model", y="time_steps", hue_order=models, data=new_df)
ax.set_ylim(40, 70)
ax.set(xlabel='', ylabel='Time steps')
plt.savefig(os.path.join(path_save, "heatmap_aggregate.pdf"))
plt.clf()
print(new_df.groupby('model').mean(),
        new_df.groupby('model').sem())


# Plotting upper triangular heatmap
model_pairs = [
    ("bd", "bd", [0,0]),
    ("bd", "up", [0, 1]),
    ("bd", "fb",[0, 2]),
    ("bd", "dc",[0, 3]),
    ("bd", "greedy", [0, 4]),
    ("up", "up", [1,1]),
    ("up","fb", [1,2,]),
    ("up","dc", [1,3]),
    ("up","greedy", [1,4]),
    ("fb", "fb", [2,2]),
    ("fb","dc", [2, 3]),
    ("fb","greedy", [2, 4]),
    ('dc', "dc", [3,3]),
    ("dc","greedy", [3, 4]),
    ('greedy', "greedy", [4,4])
] 

color_palette = sns.color_palette()
sns.set_style('ticks')
sns.set_context('talk', font_scale=1)
for recipe, map_ in itertools.product(recipes, maps):
    heat_map = np.zeros((5,5))
    heat_map[:] = np.nan
    se_map = np.zeros((5,5))
    for model1, model2, (x, y) in model_pairs:
        if split_by_env:
            data = df.loc[(df['recipe']==recipe) & (df['map']==map_) &
                (((df['model1']==model1) & (df['model2']==model2)) |
                  ((df['model1']==model2) & (df['model2']==model1)) ), :]
        else:
            data = df.loc[(((df['model1']==model1) & (df['model2']==model2)) |
                            ((df['model1']==model2) & (df['model2']==model1))), :]
        if len(data) == 0:
            print('empty data on ', model1, model2)
            continue

        heat_map[x,y] = round(data['time_steps'].mean(), 2)
        se_map[x,y] = round(data['time_steps'].std(), 2)/np.sqrt(len(data))
        print("{}, {}: {} +/- {}, {} runs".format(model1, model2,
                                     round(data['time_steps'].mean(), 2),
                                     round(data['time_steps'].std()/np.sqrt(len(data)), 2),
                                     len(data),
                                    ))

    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
    labels = (np.asarray(["{0:.1f}\n +/- {1:.1f}".format(mean, se)
                          for mean, se in zip(heat_map.flatten(),
                                                   se_map.flatten())])
             ).reshape(5,5)

    plt.figure(figsize=(9,7))
    if split_by_env:
        vmin = 0; vmax = 100
    else:
        vmin = 20; vmax = 80
    ax = sns.heatmap(heat_map,
                     cmap="Blues",
                     annot=labels,
                     annot_kws={"size": 24},
                     linewidths=.5,
                     fmt="",
                     vmin=vmin, vmax=vmax,
                    xticklabels=models,
                    yticklabels=models,
                    )
    ax.xaxis.tick_top()
    plt.yticks(np.arange(len(models))+0.5,models, rotation=90, va="center")
    ax.tick_params(axis='both', which='both', length=0)


    if split_by_env:
        # plt.title("Recipe: {}, Map: {}".format(recipe, map_))
        plt.savefig(os.path.join(path_save, "heatmap_{}_{}.pdf".format(recipe, map_)))
    else:
        plt.savefig(os.path.join(path_save, "heatmap.pdf"))
        break
