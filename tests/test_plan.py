import sys
import unittest

sys.path.append('../')
from plan import Plan
from utils.db_utils import parse_sql_query


class TestPlan(unittest.TestCase):
    def test_basic(self):
        query_tables, aliases_dict, query_conditions, reverse_aliases = parse_sql_query("""
            SELECT *
            FROM
                cast_info as ci,
                role_type as rt,
                char_name as cn,
                title as t
            WHERE cn.id = ci.person_role_id
                AND rt.role = 'actor'
                AND t.id = ci.movie_id;
        """)

        p = Plan(query_tables, aliases_dict, query_conditions, reverse_aliases)
        p.join(1, 2)
        p.join(0, 3)
        p.join(5, 4)
        query = "SELECT * FROM ((cast_info AS ci JOIN title AS t ON t.id = ci.movie_id) JOIN (role_type AS rt CROSS JOIN char_name AS cn) ON cn.id = ci.person_role_id) WHERE rt.role = 'actor'"
        assert p.sql_query().split() == query.split()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
