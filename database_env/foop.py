import numpy as np
from gym.spaces import Box, Dict

from database_env.base import DataBaseEnv


class DataBaseEnv_FOOP(DataBaseEnv):
    """
    Database environment with states and actions as in the article (https://arxiv.org/pdf/1911.11689.pdf).
    Suitable for use with RLlib.

    Attributes:
        env_config(dict): Algorithm-specific configuration data, should contain item corresponding to the DB scheme.
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        #in order to use action masks, we should use a dictionary observation space
        #(see https://docs.ray.io/en/master/rllib-models.html#variable-length-parametric-action-spaces)
        self.observation_space = Dict({
            'real_obs': Box(low = 0, high = 1, shape = (self.N_rels, self.N_cols), dtype = np.int),
            'action_mask': Box(low = 0, high = 1, shape = (len(self.actions), ), dtype = np.int),
        })
        
    def get_foop(self):
        obs = np.zeros((self.N_rels, self.N_cols))
        self._relids = []
        for node in self.plan.roots:
            rels = list(self.plan.G.nodes[node]['tables'])
            # random choice to emphasize symmetry
            rel = np.random.choice(rels)
            for r in rels:
                obs[self.rel_to_idx[rel], list(self.col_to_idx[r].values())] = 1
            self._relids.append(self.rel_to_idx[rel])
        return obs

    def get_obs(self):
        foop_obs = self.get_foop()
        valid_act = self.valid_actions()
        return {'real_obs': foop_obs, 'action_mask': list(self.valid_actions())}

    def _valid_actions_ids(self):
        valid_actions_ids = [self.action_ids[(r1, r2)] for r1 in self._relids for r2 in self._relids if r1 != r2]
        return valid_actions_ids

    def valid_actions(self):
        valid_actions = np.array([0.]*len(self.actions))
        valid_actions[self._valid_actions_ids()] = 1.
        return valid_actions
