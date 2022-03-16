import sys
import unittest
import os
import psycopg2

sys.path.append('../')
from utils.db_utils import query_to_plan


class TestParseQueryExplain(unittest.TestCase):
    def test_basic(self):
        q = """
            SELECT *
            FROM aka_name AS an,
                 cast_info AS ci,
                 name AS n
            WHERE an.person_id = n.id
              AND n.id = ci.person_id
        """

        POSTGRES_CONNECT_URL = os.environ.get(
            "POSTGRES_CONNECT_URL",
            "postgres://imdb:pwd@megatron:5678/imdb"
        )
        conn = psycopg2.connect(POSTGRES_CONNECT_URL)

        p, time = query_to_plan(q, conn, True)
        assert isinstance(time, (int, float))
        assert p.sql_query().split() == 'SELECT * FROM ((name AS n JOIN cast_info AS ci ON ci.person_id = n.id) JOIN aka_name AS an ON n.id = an.person_id)'.split()

        p2 = query_to_plan(q, conn, False)
        assert p.sql_query() == p2.sql_query()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
