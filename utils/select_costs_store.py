import logging

import psycopg2

from db_utils import get_cost_from_db, db_rollback


class SelectCostsStore():
    """
    SelectCostsStore - class for storing information about explained selects.
    Generate new table \'select_costs_store_table\' if it's not exist with columns: query, cost, exec_time.

    Parameters
    ----------
    conn: psycopg2.extensions.connection
    """

    STORE_TABLE_NAME = 'select_costs_store_table'

    def __init__(self, conn:psycopg2.extensions.connection):
        self.conn = conn
        self.logger = logging.getLogger("SelectCostsStore")
        self._init()

    def _init(self):
        """
        Initialize table for storing select costs if not exist
        """
        if not self._is_table_exist(self.STORE_TABLE_NAME):
            cur = self.conn.cursor()
            cur.execute(
                f"CREATE TABLE {self.STORE_TABLE_NAME} (id serial PRIMARY KEY, query varchar, cost real, exec_time real);"
            )
            cur.execute(f"CREATE INDEX {self.STORE_TABLE_NAME}_query_index ON {self.STORE_TABLE_NAME}(query);")
            self.conn.commit()
            cur.close()
            self.logger.info(f"Table {self.STORE_TABLE_NAME} created and db with table ready")
        else:
            self.logger.info(f"DB with table {self.STORE_TABLE_NAME} ready")

    def _is_table_exist(self, tablename:str):
        is_exist = None

        cur = self.conn.cursor()
        cur.execute("select * from information_schema.tables where table_name=%s", (tablename,))
        is_exist = bool(cur.rowcount)
        cur.close()

        return is_exist

    def get_cost(self, query:str, exec_time:bool = False):
        """
        Computer costs and optionaly exec_time for provided query or extract it from database if computed in the past.
        """
        data = self._request(query)
        if data:
            cost, computed_exec_time = data
            if exec_time:
                if computed_exec_time:
                    return (cost, computed_exec_time)
                else:
                    return self._update(query, exec_time)
            else:
                return cost
        else:
            return self._insert(query, exec_time)


    def _insert(self, query:str, exec_time:bool = False):
        """
        Insert new record with cost and optionaly exec_time to database
        """
        try:
            results = get_cost_from_db(query, self.conn, exec_time)
        except Exception as e:
            self.logger.error(f"ERROR with query: \'{query}\': {str(e)}")
            raise Exception(str(e))

        try:
            cur = self.conn.cursor()
            if exec_time:
                cost, computed_exec_time = results
                cur.execute(f"INSERT INTO {self.STORE_TABLE_NAME} (query, cost, exec_time) VALUES(%s, %s, %s)", (query, cost, computed_exec_time))
            else:
                cost = results
                cur.execute(f"INSERT INTO {self.STORE_TABLE_NAME} (query, cost) VALUES(%s, %s)", (query, cost))
            self.conn.commit()
            cur.close()

            if exec_time:
                return cost, computed_exec_time
            else:
                return cost
        except Exception as e:
            self.logger.error(str(e))
            db_rollback(self.conn)
            raise Exception(str(e))

    def _update(self, query:str, exec_time:bool = False):
        try:
            results = get_cost_from_db(query, self.conn, exec_time)
        except Exception as e:
            self.logger.error(f"ERROR with query: \'{query}\': {str(e)}")
            raise Exception(str(e))

        try:
            cur = self.conn.cursor()
            if exec_time:
                cost, computed_exec_time = results
                cur.execute(
                    f"UPDATE {self.STORE_TABLE_NAME} AS t SET cost=%s, exec_time=%s WHERE t.query=%s",
                    (cost, computed_exec_time, query)
                )
            else:
                cost = results
                cur.execute(
                    f"UPDATE {self.STORE_TABLE_NAME} AS t SET cost=%s WHERE t.query=%s",
                    (cost, query)
                )
            self.conn.commit()
            cur.close()

            if exec_time:
                return cost, computed_exec_time
            else:
                return cost
        except Exception as e:
            self.logger.error(str(e))
            db_rollback(self.conn)
            raise Exception(str(e))

    def _request(self, query:str):
        """
        Requeest cost and exec_time from database for query, return None if record for this query not exist
        """
        cur = self.conn.cursor()
        cur.execute(f"SELECT cost, exec_time FROM {self.STORE_TABLE_NAME} AS t WHERE t.query=%s", (query, ))
        data = cur.fetchone()
        cur.close()
        return data
