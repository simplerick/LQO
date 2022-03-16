import numpy as np
from gym.spaces import Box, Dict, Discrete

from database_env.foop import DataBaseEnv_FOOP
from database_env.query_encoding import DataBaseEnv_QueryEncoding


class DataBaseEnv_FOOP_QueryEncoding(DataBaseEnv_FOOP, DataBaseEnv_QueryEncoding):
    """
    Database environment with states and actions as in the article (https://arxiv.org/pdf/1911.11689.pdf)
    and encoding like NEO (http://www.vldb.org/pvldb/vol12/p1705-marcus.pdf).
    Suitable for use with RLlib.

    Attributes:
        env_config(dict): Algorithm-specific configuration data, should contain item corresponding to the DB scheme.
    """
    def __init__(self, env_config, is_join_graph_encoding=False):
        super().__init__(env_config)
        self.is_join_graph_encoding = is_join_graph_encoding

        real_obs_shape = self.N_rels * self.N_cols + self.N_cols
        if self.is_join_graph_encoding:
            real_obs_shape += self.query_encoding_size
        real_obs_shape = (real_obs_shape, )
        self.observation_space = Dict({
            'real_obs': Box(low = 0, high = 1, shape = real_obs_shape, dtype = np.int),
            'action_mask': Box(low = 0, high = 1, shape = (len(self.actions), ), dtype = np.int),
        })
    
    def get_obs(self):
        real_obs = [self.get_foop().flatten()]
        if self.is_join_graph_encoding:
            real_obs.append(self.join_graph_encoding)
        real_obs.append(self.predicate_ohe)
        
        real_obs = np.concatenate(real_obs).astype(np.int)
        return {
            'real_obs': real_obs.tolist(),
            'action_mask': self.valid_actions().astype(np.int).tolist()
        }
