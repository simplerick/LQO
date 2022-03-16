import gym
import numpy as np
import psycopg2
from gym.spaces import Discrete

from plan import Plan
from utils.db_utils import get_cost_plan
import time
import sys
sys.setrecursionlimit(10000)

import logging
LOG = logging.getLogger(__name__)


class DataBaseWithSelectQueryStoreEnv(gym.Env):
    """Base class for Data Base envs with SelectQueryStore support

    Attributes:
        psycopg_connect_url(str): string with uri for postgress
    """

    def __init__(self, psycopg_connect_url, db, return_latency=False, max_connect_retry=30000):
        self.conn = psycopg2.connect(psycopg_connect_url)
        self.psycopg_connect_url = psycopg_connect_url
        self.db = db
        self.return_latency = return_latency
        # self.sq_store = SelectCostsStore(self.conn)
        self.plan = None
        self.max_connect_retry = max_connect_retry

    def reward(self):
        def _get_cost(_retry=0):
            try:
                if _retry > 0:
                    self.conn = psycopg2.connect(self.psycopg_connect_url)
                    # self.sq_store.conn = self.conn
                cost = get_cost_plan(self.plan, self.conn,
                                     self.db, self.return_latency)
                return cost
            except psycopg2.errors.DiskFull:
                # SELECT pg_terminate_backend(pg_state_activity.pid) FROM pg_state_activity WHERE pg_state_activity.datname = 'imdb';
                LOG.error(f'Disk is full. Query id: {self.query_id}')
                return None
            except psycopg2.errors.QueryCanceled:
                # SELECT pg_terminate_backend(pg_state_activity.pid) FROM pg_state_activity WHERE pg_state_activity.datname = 'imdb';
                LOG.warning(f'Statement timeout.')
                return 3*10**5
            except Exception as e:
                LOG.warning(f"{e}; Retry {_retry}")
                if _retry < self.max_connect_retry:
                    time.sleep(10)
                    return _get_cost(_retry+1)
                else:
                    LOG.error(
                        f'Max attempts has been reached. Query {self.query_id} is skipped.')
                    return None
        if (self.plan and self.plan.is_complete):
            return _get_cost(0)
        else:
            return None


class DataBaseEnv(DataBaseWithSelectQueryStoreEnv):
    """Custom Environment that follows gym interface and interacts with PostgreSQL
       Suitable for use with RLlib.

       Attributes:
           env_config(dict): Algorithm-specific configuration data,
           should contain item corresponding to the DB scheme.
    """

    def __init__(self, env_config):
        super().__init__(env_config['psycopg_connect_url'], env_config['db'],
                         env_config.get('return_latency', False))
        self.scheme = env_config['scheme']
        self.db_data = env_config['db_data']
        self.rel_to_idx = {rel: i for i, rel in enumerate(
            self.scheme)}  # idx = obj[rel]
        self.rels = list(self.rel_to_idx.keys())
        self.N_rels = len(self.rels)
        self.col_to_idx = dict()  # idx = obj[rel][col]
        self.N_cols = 0
        self.cols = []
        for rel in self.scheme:
            self.col_to_idx[rel] = {}
            for col in self.scheme[rel]:
                self.col_to_idx[rel][col] = self.N_cols
                self.N_cols += 1
                self.cols.append((rel, col))

        self.actions = [(r1, r2) for r1 in range(self.N_rels)
                        for r2 in range(self.N_rels) if r1 != r2]
        self.action_ids = {a: i for i, a in enumerate(self.actions)}
        self.action_space = Discrete(len(self.actions))

    def get_obs(self):
        """
        Constructs an observation based on the plan.
        """
        raise NotImplementedError

    def from_action(self, action):
        """
        Retrieve relations to join from action.
        """
        r1, r2 = self.actions[action]
        n1 = self.plan.rel_to_node(self.rels[r1])
        n2 = self.plan.rel_to_node(self.rels[r2])
        return n1, n2

    def valid_actions(self):
        """
        Computes valid actions.
        Any nodes in plan can be joined
        """
        relids = self.plan.query_tables
        valid_actions_ids = [
            self.action_ids[(r1, r2)] for r1 in relids for r2 in relids if r1 != r2]
        valid_actions = np.zeros(len(self.actions))
        valid_actions[valid_actions_ids] = 1.
        return valid_actions

    @property
    def is_done(self):
        return self.plan.is_complete

    def step(self, action):
        self.current_step += 1
        tables_to_join = self.from_action(action)
        self.plan.join(*tables_to_join)
        return self.get_obs(), self.reward(), self.is_done, {}

    def reset(self, idx=None):
        if isinstance(idx, str):
            self.query_id = idx
        elif isinstance(idx, int):
            self.query_id = list(self.db_data.keys())[idx]
        else:
            self.query_id = np.random.choice(list(self.db_data.keys()))
        self.plan = Plan(*self.db_data[self.query_id])
        self.current_step = 0

    def render(self):
        return self.plan.render()

    def find_cost(self, p):
        return list(p.G.nodes(data=True))[-1][-1]['cost'][0]
