import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle
import sys
sys.path.append("../..")
import recipe_planner


recipes = [
        "tomato",
        "tl",
        "salad"
    ]
total_num_subtasks = {
        "tomato": 3,
        "tl": 6,
        "salad": 5
    }
models = [
       "_model1-bd_model2-bd",
       "_model1-up_model2-up",
       "_model1-fb_model2-fb",
       "_model1-dc_model2-dc",
       "_model1-greedy_model2-greedy",
    ]
model_key = {
    "_model1-bd_model2-bd": "BD (ours)",
    "_model1-up_model2-up": "UP",
    "_model1-fb_model2-fb": "FB",
    "_model1-dc_model2-dc": "D&C",
    "_model1-greedy_model2-greedy": "Greedy",
}
maps = [
        "full-divider",
        "open-divider",
        "partial-divider"
        ]
seeds = range(1,10)
agents = ['agent-1', 'agent-2', 'agent-3', 'agent-4']
agents2_optimal = {
    "open-divider": {"tomato": 15, "tl": 25, "salad": 24},
    "partial-divider": {"tomato": 17, "tl": 31, "salad": 21},
    "full-divider": {"tomato": 17, "tl": 31, "salad": 21}
}
agents3_optimal = {
    "open-divider": {"tomato": 12, "tl": 22, "salad": 15},
    "partial-divider": {"tomato": 12, "tl": 22, "salad": 16},
    "full-divider": {"tomato": 13, "tl": 24, "salad": 19}
}
time_steps_optimal = {2: agents2_optimal, 3: agents3_optimal}

ylims = {
    'time_steps': [0, 100],
    'shuffles': [0, 55],
    'priors': [0, 100],
    'completion': [0, 1],
}

ylabels = {
    'time_steps': 'Time',
    'completion': 'Completion',
    'shuffles': 'Shuffles',
    'priors': 'Time',
    'completion': 'Completion'
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="for parsing")
    parser.add_argument("--num-agents", type=int, default=2, help="number of agents")
    parser.add_argument("--stats", action="store_true", default=False, help="run and print out summary statistics")
    parser.add_argument("--time-steps", action="store_true", default=False, help="make graphs for time_steps")
    parser.add_argument("--completion", action="store_true", default=False, help="make graphs for completion")
    parser.add_argument("--shuffles", action="store_true", default=False, help="make graphs for shuffles")
    parser.add_argument("--legend", action="store_true", default=False, help="make legend alongside graphs")
    return parser.parse_args()


def run_main():
    #path_pickles = '/Users/custom/path/to/pickles'
    path_pickles = os.path.join(os.getcwd(), 'pickles')
    #path_save = '/Users/custom/path/to/save/to'
    path_save = os.path.join(os.getcwd(), 'graphs_agents{}'.format(arglist.num_agents))
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if arglist.stats:
        compute_stats(path_pickles, arglist.num_agents)
        return

    if arglist.time_steps:
        key = 'time_steps'
    elif arglist.completion:
        key = 'completion'
    elif arglist.shuffles:
        key = 'shuffles'
    else:
        return
    df = import_data(key, path_pickles, arglist.num_agents)
    print('done loading pickle data')
    plot_data(key, path_save, df, arglist.num_agents, legend=arglist.legend)


def compute_stats(path_pickles, num_agents):
    for model in models:
        num_shuffles = []; num_timesteps = []; num_collisions = []; frac_completed = []
        for recipe, map_, seed in itertools.product(recipes, maps, seeds):
            fname = '{}_{}_agents{}_seed{}{}.pkl'.format(map_, recipe, num_agents, seed, model)
            if os.path.exists(os.path.join(path_pickles, fname)):
                try:
                    data = pickle.load(open(os.path.join(path_pickles, fname), "rb"))
                except EOFError:
                    continue
                shuffles = get_shuffles(data, recipe)   # dict of 2 numbers
                num_shuffles += shuffles.values()
                num_timesteps.append(get_time_steps(data, recipe))
                if data['was_successful']:
                    num_collisions.append(get_collisions(data, recipe))
                frac_completed.append(get_frac_completed(data, recipe))
            else:
                print('no file:', fname)
                continue

        print('{}   time steps: {:.3f} +/- {:.3f}'.format(model_key[model], np.mean(np.array(num_timesteps)), np.std(np.array(num_timesteps))/np.sqrt(len(num_timesteps))))
        print('     frac_completed: {:.3f} +/- {:.3f}'.format(np.mean(np.array(frac_completed)), np.std(np.array(num_collisions))/np.sqrt(len(frac_completed))))
        print('     collisions: {:.3f} +/- {:.3f}'.format(np.mean(np.array(num_collisions)), np.std(np.array(num_collisions))/np.sqrt(len(num_collisions))))
        print('     shuffles: {:.3f} +/- {:.3f}'.format(np.mean(np.array(num_shuffles)), np.std(np.array(num_collisions))/np.sqrt(len(num_shuffles))))


def import_data(key, path_pickles, num_agents):
    df = list()

    for recipe, model, map_, seed in itertools.product(recipes, models, maps, seeds):
        info = {
            "map": map_,
            "seed": seed,
            "recipe": recipe,
            'model': model_key[model],
            "dummy": 0
        }

        # LOAD IN FILE
        fname = '{}_{}_agents{}_seed{}{}.pkl'.format(map_, recipe, num_agents, seed, model)
        if os.path.exists(os.path.join(path_pickles, fname)):
            try:
                data = pickle.load(open(os.path.join(path_pickles, fname), "rb"))
            except:
                print("trouble loading: {}".format(fname))
        else:
            print('no file:', fname)
            continue

        # TIME STEPS
        if key == 'time_steps':
            time_steps = get_time_steps(data, recipe)
            print("{}: {}".format(fname, time_steps))
            df.append(dict(time_steps = time_steps, **info))

        # COMPLETION
        elif key == 'completion':
            for t in range(100):
                n = get_completion(data, recipe, t)
                df.append(dict({'t': t-1, 'n': n}, **info))

        # SHUFFLES
        elif key == 'shuffles':
            shuffles = get_shuffles(data, recipe)   # a dict
            df.append(dict(shuffles = np.mean(np.array(list(shuffles.values()))), **info))
            # for agent in agents:
            #     info['agent'] = agent
            #     df.append(dict(shuffles = shuffles[agent], **info))

    return pd.DataFrame(df)

def get_time_steps(data, recipe):
    try:
        # first timestep at which required number of recipe subtasks has been completed
        # using this instead of total length bc of termination bug
        return data['num_completed_subtasks'].index(total_num_subtasks[recipe])+1
    except:
        return 100

def get_completion(data, recipe, t):
    df = list()
    completion = data['num_completed_subtasks']
    try:
        end_indx = completion.index(total_num_subtasks[recipe])+1
        completion = completion[:end_indx]
    except:
        end_indx = None
    if len(completion) < 100:
        completion += [data['num_completed_subtasks_end']]*(100-len(completion))
    assert len(completion) == 100
    return completion[t]/total_num_subtasks[recipe]

def get_shuffles(data, recipe):
    # recipe isn't needed but just for consistency
    # returns a dict, although we only use average of all agents
    shuffles = {}
    for agent in data['actions'].keys():
        count = 0
        actions = data['actions'][agent]
        holdings = data['holding'][agent]
        for t in range(2, len(holdings)):
            # count how many negated the previous action
            # redundant movement
            if holdings[t-2] == holdings[t-1] and holdings[t-1] == holdings[t]:
                net_action = np.array(actions[t-1]) + np.array(actions[t])
                redundant = (net_action == [0, 0])
                if redundant.all() and actions[t] != (0, 0):
                    count += 1
                    # print(agent, t, actions[t-1], holdings[t-1], actions[t], holdings[t], actions[t+1], holdings[t+1])
            # redundant interaction
            elif holdings[t-2] != holdings[t-1] and holdings[t-2] == holdings[t]:
                redundant = (actions[t-1] == actions[t] and actions[t] != (0, 0))
                if redundant:
                    count += 1
                    # print(agent, t, actions[t-1], holdings[t-1], actions[t], holdings[t], actions[t+1], holdings[t+1])
        shuffles[agent] = count
    return shuffles

def plot_data(key, path_save, df, num_agents, legend=False):
    print('generating {} graphs'.format(key))
    hue_order = [model_key[l] for l in models]
    color_palette = sns.color_palette()
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=1)

    for i, recipe in enumerate(recipes):
        for j, map_ in enumerate(maps):
            data = df.loc[(df['map']==map_) & (df['recipe']==recipe), :]
            if len(data) == 0:
                print('empty data on ', (recipe, map_))
                continue

            plt.figure(figsize=(3,3))

            if key == 'completion':
                # plot ours last
                hue_order = hue_order[1:] + [hue_order[0]]
                color_palette = sns.color_palette()[1:5] + [sns.color_palette()[0]]
                ax = sns.lineplot(x = 't', y = 'n', hue="model", data=data,
                    linewidth=5, legend=False, hue_order=hue_order, palette=color_palette)
                plt.xlabel('Steps')
                plt.ylim([0, 1]),
                plt.xlim([0, 100])
            else:
                hue_order = hue_order[1:] + [hue_order[0]]
                color_palette = sns.color_palette()[1:5] + [sns.color_palette()[0]]
                sns.barplot(x='dummy', y=key, hue="model", data=data, hue_order=hue_order,\
                                palette=color_palette, ci=68).set(
                    xlabel = "",
                    xticks = [],
                    ylim = ylims[key],
                )
            plt.legend('')
            plt.gca().legend().set_visible(False)
            sns.despine()
            plt.tight_layout()

            plt.ylabel(ylabels[key])

            if recipe != 'tomato' and key == 'priors':
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.ylabel('')

            if key == 'time_steps' or key == 'priors':
                plt.axhline(y = time_steps_optimal[num_agents][map_][recipe], ls='--', color='black')

            plt.savefig(os.path.join(path_save, "{}_{}_{}.png".format(key, recipe, map_)))
            plt.close()

            print('   generated graph for {}, {}'.format(recipe, map_))

    # Make Legend
    if arglist.legend:
        plt.figure(figsize=(10,10))
        if key == 'completion':
            sns.barplot(x = 't', y = 'n', hue="model", data=data, hue_order=hue_order, palette=color_palette, ci=68).set()
        else:
            sns.barplot(x='dummy', y=key, hue="model", data=data, hue_order=hue_order, palette=color_palette, ci=68).set(
                xlabel = "", xticks = [], ylim = [0, 1000])
        legend = plt.legend(frameon=False)
        legend_fig = legend.figure
        legend_fig.canvas.draw()
        # bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
        # legend_fig.savefig(os.path.join(path_save, 'legend.pdf'), dpi="figure", bbox_inches=bbox)
        legend_fig.savefig(os.path.join(path_save, '{}_legend_full.png'.format(key)), dpi="figure")
        plt.close()



if __name__ == "__main__":
    arglist = parse_arguments()
    run_main()


