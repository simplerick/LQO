import re
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import numpy as np
import psycopg2
from gym.spaces import Box, Dict

from database_env.base import DataBaseEnv


class DataBaseEnv_QueryEncoding(DataBaseEnv):
    """Database environment with states and actions as in the NEO paper
    (http://www.vldb.org/pvldb/vol12/p1705-marcus.pdf)
    Suitable for use with RLlib.

    Attributes:
        env_config(dict): Algorithm-specific configuration data, should contain item corresponding to the DB scheme.
    """

    def __init__(self, env_config):
        super().__init__(env_config)
        self.query_encoding_size = int(self.N_rels * (self.N_rels - 1) / 2)
        self.observation_space = Dict({
            'real_obs': Box(low=0, high=1, shape=(self.query_encoding_size + self.N_cols, ), dtype=np.int),
            'action_mask': Box(low=0, high=1, shape=(len(self.actions), ), dtype=np.int),
        })
        self._initial_query = None
        self.cardinalities = env_config.get('cardinalities')
        self.condition_selectivity = env_config.get('condition_selectivity')
        self.tables_features = env_config.get('tables_features')
        self.index_features = env_config.get('index_features')

    def compute_cardinalities(env_config):
        conn = psycopg2.connect(env_config["psycopg_connect_url"])
        cursor = conn.cursor()
        env_config["cardinalities"] = {}
        for q, info in env_config["db_data"].items():
            card = defaultdict(int)
            d = defaultdict(list)
            for cond in info[2]:
                if len(cond['names']) == 1:
                    d[cond['names'][0]].append(cond['condition'])
            for table in info[1].keys():
                conds = d[table]
                if len(conds) == 0:
                    d[table] = 1
                    continue
                query1 = f"SELECT count(*) FROM {info[1][table]} AS {table} WHERE {' AND '.join(conds)};"
                query2 = f"SELECT count(*) FROM {info[1][table]}"
                cursor.execute(query1)
                num_after = cursor.fetchall()[0][0]
                # cursor.execute(query2)
                # num_before = cursor.fetchall()[0][0]
                # d[table] = num_after/(num_before+1e-10)
                card[table] = np.log(num_after)
            env_config["cardinalities"][info[3]] = card
        cursor.close()
        conn.close()

    def compute_condition_selectivities(env_config):
        conn = psycopg2.connect(env_config["psycopg_connect_url"])
        cursor = conn.cursor()
        env_config["condition_selectivity"] = {}
        for q, info in env_config["db_data"].items():
            condition_selectivity = defaultdict(int)
            for cond in info[2]:
                if len(cond['names']) == 1:
                    als = cond['names'][0]
                    quer1 = f"EXPLAIN (FORMAT JSON) SELECT * FROM {info[1][als]} AS {als}"
                    quer2 = f"EXPLAIN (FORMAT JSON) SELECT * FROM {info[1][als]} AS {als} WHERE {cond['condition']}"
                    cursor.execute(quer1)
                    num_rows_1 = cursor.fetchall()[0][0][0]['Plan']['Plan Rows']
                    cursor.execute(quer2)
                    num_rows_2 = cursor.fetchall()[0][0][0]['Plan']['Plan Rows']
                    condition_selectivity[cond['condition']] = num_rows_2/(num_rows_1 + 1e-6)
            env_config["condition_selectivity"][info[3]] = condition_selectivity
        cursor.close()
        conn.close()

    def get_predicates_encoding(self,plan):
        if self.condition_selectivity is None:
            return self.get_predicates_ohe(plan)
        else:
            return self.get_predicates_selectivity(plan)

    def get_predicates_selectivity(self, plan):
        column_preicates_vector = np.zeros(self.N_cols)
        selectivities = self.condition_selectivity[plan.initial_query]
        for aliases, conditions in plan._query_join_conditions.items():
            if len(aliases) != 1:
                continue
            alias = next(iter(aliases))
            tab_name = plan.alias_to_table[alias]
            for condition in conditions:
                col_name = self._parse_condition(condition)[0][-1]
                idx = self.col_to_idx[tab_name][col_name]
                # # one hot
                # column_preicates_vector[idx] = 1
                # selectivity
                column_preicates_vector[idx] += selectivities[condition]
        return column_preicates_vector


    def get_predicates_ohe(self, plan):
        column_preicates_vector = np.zeros(self.N_cols)
        for condition in plan.query_join_conditions:
            predicates = self._parse_condition(condition['condition'])
            for alias, col_name in predicates:
                tab_name = plan.alias_to_table[alias]
                idx = self.col_to_idx[tab_name][col_name]
                column_preicates_vector[idx] = 1
        return column_preicates_vector


    def get_join_graph_encoding(self, plan):
        join_graph_matrix = np.zeros((self.N_rels, self.N_rels))
        for tabs in plan._query_join_conditions.keys():
            if len(tabs) == 2:
                tab1, tab2 = tabs
                tab1, tab2 = plan.alias_to_table[tab1], plan.alias_to_table[tab2]
                tab1_idx, tab2_idx = self.rel_to_idx[tab1], self.rel_to_idx[tab2]
                join_graph_matrix[[tab1_idx, tab2_idx],
                                  [tab2_idx, tab1_idx]] = 1
        return join_graph_matrix[np.triu_indices(self.N_rels, 1)]

    def get_obs(self, plan):
        if (plan.initial_query != self._initial_query):
            self._initial_query = plan.initial_query
            self.compute_query_enc(plan)
        features = [self.join_graph_encoding, self.predicate_ohe]
        if self.tables_features is not None:
            features.append(self.tables_features.flatten())
            features.append(self.index_features.flatten())
        return np.concatenate(features)

    def compute_query_enc(self, plan):
        self.predicate_ohe = self.get_predicates_encoding(plan)
        self.join_graph_encoding = self.get_join_graph_encoding(plan)

    def _parse_condition(self, condition: str) -> ((str, str), (str, str)):
        """Helper for parsing query condition
        Will return (table_name_1, column_name_1), (table_name_2, column_name_2)
        """
        return re.findall(r'([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)', condition)

    def compute_data_driven_features(env_config):
        conn = psycopg2.connect(env_config["psycopg_connect_url"])
        cursor = conn.cursor()
        cursor.execute('''
            SELECT seq_scan, seq_tup_read, n_tup_ins, n_tup_upd, n_tup_del,
            n_tup_hot_upd, n_live_tup, n_dead_tup, n_mod_since_analyze, vacuum_count, autovacuum_count
            FROM pg_stat_user_tables;
            ''')
        tables_features = cursor.fetchall()
        cursor.close()

        tables_features = np.array(tables_features)
        tables_features[tables_features == None] = 0

        dd_table_features_scaler = StandardScaler()
        dd_table_features_scaler.fit(tables_features)

        env_config['tables_features'] = dd_table_features_scaler.transform(
            tables_features
        )

        cursor = conn.cursor()
        cursor.execute("""
            SELECT idx_scan, idx_tup_read, idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname='public';
            """)
        index_features = cursor.fetchall()
        cursor.close()

        index_features = np.array(index_features)
        index_features[index_features == None] = 0

        dd_index_features_scaler = StandardScaler()
        dd_index_features_scaler.fit(index_features)

        env_config['index_features'] = dd_index_features_scaler.transform(
            index_features
        )
        conn.close()
