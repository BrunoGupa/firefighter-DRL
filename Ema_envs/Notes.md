in gym-cellular-automata/gym_cellular_automata/forest_fire/bulldozer/bulldozer.py
we fix the variables 

* t_move = 0.05 
* t_act = 0.15
* pos_bull = [nrows-2,ncols-2] (not yet implemented)

I modify the function:

- def _award()
    return 0

I added the fire in:

self._effects = {self._tree: self._empty,
                         self._fire: self._empty}  # Substitution Effect


The environment takes t_move + t_act at each step.
if the sum is greater than 1 then the CA actualizes and then the fire propagates.

I use wrapper in the observation to:
* convert into a One-Hot encoding.
* Change the observation to MultiBinary

We add the following RewardWrapper:
the total of tree at the end of the episode
