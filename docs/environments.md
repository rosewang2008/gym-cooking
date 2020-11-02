
## Environments and Recipes

Please refer to our [Overcooked Design Document](design.md) for more information on environment/task customization and observation/action spaces.

Our environments and recipes are combined into one argument. Here are the environment

# Full Divider

A single 7x7 room where there's a long divider confining agents to one half of the space. 

To run this environment on our available recipes (tomato, tomato-lettuce, and salad), run `main.py` with the following argument:

`python3 main.py --level full-divider_tomato ...other args...`

`python3 main.py --level full-divider_tl ...other args...`

`python3 main.py --level full-divider_salad ...other args...`

<p align="center">
<img src="/images/full.png" width=300></img>
</p>


# Partial Divider

A single 7x7 room where there's a long divider. Agents can still move through the entire space. 

To run this environment on our available recipes (tomato, tomato-lettuce, and salad), run `main.py` with the following argument:

`python3 main.py --level partial-divider_tomato ...other args...`

`python3 main.py --level partial-divider_tl ...other args...`

`python3 main.py --level partial-divider_salad ...other args...`

<p align="center">
<img src="/images/partial.png" width=300></img>
</p>

# Open Divider 

A single 7x7 room where there's no divider. Agents can move through the entire space. 

To run this environment on our available recipes (tomato, tomato-lettuce, and salad), run `main.py` with the following argument:

`python3 main.py --level open-divider_tomato ...other args...`

`python3 main.py --level open-divider_tl ...other args...`

`python3 main.py --level open-divider_salad ...other args...`

<p align="center">
<img src="/images/open.png" width=300></img>
</p>
