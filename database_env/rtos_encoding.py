import re
from operator import xor

import numpy as np
from gym import spaces
from gym.spaces import Box, Dict

from database_env.query_encoding import DataBaseEnv_QueryEncoding


class DataBaseEnv_RTOSEncoding(DataBaseEnv_QueryEncoding):
    """Database environment with states and actions as
    in the RTOS paper (http://da.qcri.org/ntang/pubs/icde20jos.pdf)
    Suitable for use with RLlib.

    Attributes:
        env_config(dict): Algorithm-specific configuration data,
            should contain item corresponding to the DB scheme.
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        self.minmax_scheme = env_config['minmax_scheme']
        self.query_encoding_size = int(self.N_rels * (self.N_rels - 1) / 2)
        self.observation_space = Dict({
            'real_obs': Box(
                low=0, high=1,
                shape=(self.query_encoding_size + self.N_cols * 4, ),
                dtype=np.int
            ),
            'action_mask': Box(
                low=0, high=1,
                shape=(len(self.actions), ),
                dtype=np.int
            ),
        })

    def _norm_condition_value(self, v, c_min, c_max):
        return (v - c_min) / (c_max - c_min)

    def _fill_numeric_column_feature_vector(self, tab_name, col_name, sign, v):
        vec = [0, 0, 0, 0]
        c_min = self.minmax_scheme[tab_name][col_name]['min']
        c_max = self.minmax_scheme[tab_name][col_name]['max']
        if sign == '=':
            if v < c_min or v > c_max:
                vec[1] = 1
            else:
                vec[1] = self._norm_condition_value(v, c_min, c_max) + 1
        elif sign in ['>', '>=']:
            if v >= c_max:
                vec[2] = 1
            elif v < c_min:
                vec[2] = 0
            else:
                vec[2] = self._norm_condition_value(v, c_min, c_max)
        elif sign in ['<', '<=']:
            if v <= c_min:
                vec[3] = 1
            elif v > c_max:
                vec[3] = 0
            else:
                vec[3] = 1 - self._norm_condition_value(v, c_min, c_max)
        else:
            raise Exception(f'worong comparsion sign in condition {sign}')

        return vec

    def get_column_representation(self):
        feature_vectors = np.zeros((self.N_cols, 4))

        for condition in self.plan.query_join_conditions:
            parsed_cond = self._parse_condition(condition['condition'])
            if parsed_cond:
                p1, sign, p2 = parsed_cond
            else:
                continue

            if xor((not isinstance(p1, tuple)), (not isinstance(p2, tuple))):
                if isinstance(p1, tuple):
                    tab_name, col_name = p1
                    v = p2
                else:
                    tab_name, col_name = p2
                    v = p1
                tab_name = self.plan.alias_to_table.get(tab_name, tab_name)

                column_idx = self.cols.index((tab_name, col_name))
                if self.minmax_scheme[tab_name][col_name]['numeric']:
                    feature_vectors[column_idx] = \
                        self._fill_numeric_column_feature_vector(tab_name, col_name, sign, float(v))
                else:
                    feature_vectors[column_idx][1] = 1
            else:
                if isinstance(p1, tuple):
                    tab_name, col_name = p1
                elif isinstance(p2, tuple):
                    tab_name, col_name = p2
                else:
                    continue # Bouth condition statement is not table.column

                tab_name = self.plan.alias_to_table.get(tab_name, tab_name)
                column_idx = self.cols.index((tab_name, col_name))
                feature_vectors[column_idx][0] = 1

        return feature_vectors

    def get_single_table_matrix_representation(self, table_name):
        table_representation = np.zeros((len(self.scheme[table_name]), 4))
        for i, col_name in enumerate(self.scheme[table_name]):
            column_representation_idx = self.cols.index((table_name, col_name))
            table_representation[i] = self.column_represenation[column_representation_idx].copy()
        return table_representation

    def get_tables_matrix_representation(self, table_names):
        representations = []
        for tab_name in table_names:
            representations.append(self.get_single_table_matrix_representation(tab_name))
        return np.concatenate(representations, axis=0)

    def generate_join_tree_rtos_features(self):
        query_join_conditions = self.plan._query_join_conditions.copy()

        def _generate_tree_node_feature(node):
            successors = self.plan.G[node]
            if len(successors) == 0:
                tables = self.plan.G.nodes[node]['tables']
                feature = np.zeros((4, 4))
                feature[0] = self.get_tables_matrix_representation(tables).max(axis=0)
                self.plan.G.nodes[node]['feature'] = feature
                return None

            l, r = successors
            _generate_tree_node_feature(l)
            _generate_tree_node_feature(r)

            l_tables = self.plan.G.nodes[l]['tables']
            r_tables = self.plan.G.nodes[r]['tables']

            if self.plan.G.nodes[node].get('condition'):
                condition = self.plan.G.nodes[node].get('condition')
                conditions = [condition]
                if condition in query_join_conditions:
                    query_join_conditions.pop(condition)
            else:
                conditions = []
                for r_t in r_tables:
                    for l_t in l_tables:
                        pair = frozenset((l_t, r_t))
                        if pair in query_join_conditions:
                            conditions.append(query_join_conditions.pop(pair))

            if len(conditions) > 0:
                condition = conditions[-1]
                results = re.findall(
                    r'([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)( [\=\>\<]+ )+([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)',
                    condition
                )
                tab1, col1, op, tab2, col2 = results[0]
                tab1 = self.plan.alias_to_table.get(tab1, tab1)
                tab2 = self.plan.alias_to_table.get(tab2, tab2)

                col_repr_idx1 = self.cols.index((tab1, col1))
                col_repr_idx2 = self.cols.index((tab2, col2))
                self.plan.G.nodes[node]['feature'] = np.concatenate(
                    [
                        [self.column_represenation[col_repr_idx1]],
                        [self.get_tables_matrix_representation(l_tables).max(axis=0)],
                        [self.get_tables_matrix_representation(r_tables).max(axis=0)],
                        [self.column_represenation[col_repr_idx2]],
                    ],
                    axis=0,
                )


        _generate_tree_node_feature(list(self.plan.roots)[0])

    def get_obs(self):
        return {
            'real_obs': np.concatenate([
                self.join_graph_encoding,
                self.column_represenation.flatten()
            ]).astype(np.int).tolist(),
            'action_mask': self.valid_actions().astype(np.int).tolist()
        }

    def reset(self, idx=None):
        self._reset(idx)
        self.column_represenation = self.get_column_representation()
        self.join_graph_encoding = self.get_join_graph_encoding()
        # self.generate_join_tree_rtos_features()
        return self.get_obs()

    def _parse_condition(self, condition: str):
        """Helper for parsing query condition
        For example for `table_name_1.column_name_1 = 1`
        will return ((table_name_1, column_name_1), '=', 12)
        """
        comparsion_regexp = r'\ *([\<\=\>]+)\ *'
        predicate_regexp = r'^([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)$'
        preds = re.split(comparsion_regexp, condition)

        def _extract_predicate(pred):
            parsed_pred = re.findall(predicate_regexp, pred)
            if parsed_pred:
                parsed_pred = parsed_pred[0]
            else:
                parsed_pred = pred
            return parsed_pred

        if len(preds) > 1:
            return _extract_predicate(preds[0]), preds[1], _extract_predicate(preds[-1])
        else:
            return None

