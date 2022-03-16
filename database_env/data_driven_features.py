from .base import DataBaseEnv
from sklearn.preprocessing import StandardScaler
import psycopg2
import numpy as np


class PGDataDrivenFeatures(DataBaseEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.dd_table_features_scaler = None
        self.dd_index_features_scaler = None

    def get_data_driven_features(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT seq_scan, seq_tup_read, n_tup_ins, n_tup_upd, n_tup_del,
            n_tup_hot_upd, n_live_tup, n_dead_tup, n_mod_since_analyze, vacuum_count, autovacuum_count
            FROM pg_stat_user_tables;
            ''')
        tables_features = cursor.fetchall()
        cursor.close()

        tables_features = np.array(tables_features)
        tables_features[tables_features == None] = 0

        if self.dd_table_features_scaler is None:
            self.dd_table_features_scaler = StandardScaler()
            self.dd_table_features_scaler.fit(tables_features)

        self.tables_features = self.dd_table_features_scaler.transform(
            tables_features
        )

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT idx_scan, idx_tup_read, idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname='public';
            """)
        index_features = cursor.fetchall()
        cursor.close()

        index_features = np.array(index_features)
        index_features[index_features == None] = 0

        if self.dd_index_features_scaler is None:
            self.dd_index_features_scaler = StandardScaler()
            self.dd_index_features_scaler.fit(index_features)

        self.index_features = self.dd_index_features_scaler.transform(
            index_features
        )
