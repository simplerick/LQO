import unittest
import os
import pickle
import sys

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

sys.path.append('../')
from database_env import (DataBaseEnv_FOOP, DataBaseEnv_FOOP_QueryEncoding)
from rllib_model import CustomMaskedActModel


class TestDataBaseEnv(unittest.TestCase):
    def setUp(self) -> None:
        ray.shutdown()
        ray.init()

        with open('./data/data_parsed.pickle', 'rb') as f:
            self.data_parsed = pickle.load(f)

        with open('./data/init_schema.pickle', 'rb') as f:
            self.init_schema = pickle.load(f)

        self.POSTGRES_CONNECT_URL = os.environ.get(
            "POSTGRES_CONNECT_URL",
            "postgres://imdb:pwd@megatron:5678/imdb"
        )

        self.config = {
            "model": {
                "custom_model": CustomMaskedActModel,
                # "custom_model_config": {'true_obs_shape': true_obs_shape},
                # "fcnet_hiddens": [512, MAX_LEN_ACT]
            },
            "num_gpus": 0,
            "num_workers": 0,
            "framework": "torch",
            "env_config": {
                "psycopg_connect_url" : self.POSTGRES_CONNECT_URL,
                "scheme": self.init_schema,
                "db_data": self.data_parsed
            },
            "explore": True,
            "preprocessor_pref": None,
            "clip_actions": False
        }

    def tearDown(self) -> None:
        ray.shutdown()

    def test_database_foop_env(self):
        ENV_NAME = "DataBaseEnv_FOOPv1"
        register_env(ENV_NAME, lambda config: DataBaseEnv_FOOP(config))

        FOOP_SHAPE_0 = len(self.init_schema.keys())
        FOOP_SHAPE_1 = sum(len(v) for v in self.init_schema.values())
        MAX_LEN_ACT = FOOP_SHAPE_0*(FOOP_SHAPE_0-1)

        self._config_and_run((FOOP_SHAPE_0, FOOP_SHAPE_1), MAX_LEN_ACT, ENV_NAME)

    def test_database_foop_query_encoding_with_join_graph(self):
        ENV_NAME = "DataBaseEnv_FOOP_QueryEncodingv1"
        register_env(ENV_NAME, lambda config: DataBaseEnv_FOOP_QueryEncoding(config, True))

        N_RELS = len(self.init_schema.keys())
        N_COLS = sum(len(v) for v in self.init_schema.values())
        MAX_LEN_ACT = N_RELS*(N_RELS-1)

        true_obs_shape = (int(N_RELS * N_COLS + N_RELS * (N_RELS - 1) / 2 + N_COLS), )

        self._config_and_run(true_obs_shape, MAX_LEN_ACT, ENV_NAME)

    def _config_and_run(self, true_obs_shape, max_len_act, env_name):
        self.config["model"]["custom_model_config"] = {"true_obs_shape": true_obs_shape}
        self.config["model"]["fcnet_hiddens"] = [8, max_len_act]

        agent = ppo.PPOTrainer(self.config, env=env_name)
        agent.train()


if __name__ == "__main__":
    unittest.main()
