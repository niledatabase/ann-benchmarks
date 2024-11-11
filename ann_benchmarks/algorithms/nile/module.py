import pgvector.psycopg
import psycopg
from psycopg import sql
import numpy
import psutil
from typing import Optional
from ..base.module import BaseANN


class Nile(BaseANN):
    BATCH_SIZE = 1000 # need to tune this
    IS_TENANT_AWARE = True # TODO: make this configurable
    TENANT_ID = "00000000-0000-0000-0000-000000000000" # TODO: Support multiple tenants
    
    def __init__(self, metric: str, connection_string: str, m: int, ef_construction: int, existing_table: bool = False):
        self._metric = metric
        self._connection_string = connection_string
        self._m = m
        self._ef_construction = ef_construction
        self._existing_table = existing_table
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id, %(query_embedding)s::vector<=>embedding as distance FROM items ORDER BY distance LIMIT {limit}"
        elif metric == "euclidean":
            self._query = "SELECT id, %(query_embedding)s::vector<->embedding as distance FROM items ORDER BY distance LIMIT {limit}"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
    def _create_table(self, cur, dimensions):
        if self.IS_TENANT_AWARE:
            self._create_tenant_aware_table(cur, dimensions)
        else:
            self._create_shared_table(cur, dimensions)

    def _create_shared_table(self, cur, dimensions):
        print("creating table...")
        cur.execute("DROP INDEX IF EXISTS items_embedding_idx") # needed because Nile doesn't support CASCADE
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % dimensions)
        
    def _create_tenant_aware_table(self, cur, dimensions):
        print("creating table...")
        try:
            cur.execute("DROP INDEX IF EXISTS items_embedding_idx") # needed because Nile doesn't support CASCADE
        except Exception as e:
            print(e) # ignore if index doesn't exist, this is a workaround for THE-2303
            
        try:
            cur.execute("DROP TABLE IF EXISTS items")
            cur.execute("CREATE TABLE items (id int, tenant_id uuid, embedding vector(%d))" % dimensions) 
        except Exception as e:
            print(e)
            raise e
            
        try:
            cur.execute("INSERT INTO tenants(id, name) VALUES (%s, %s)", (self.TENANT_ID, "default"))
        except Exception as e:
            print(e) # ignore if tenant already exists

    def _insert_data(self, cur, conn, X):
        print("inserting data...")
        print(f"Loading {X.shape[0]} embeddings into table")
        
        if self.IS_TENANT_AWARE:
            query = "INSERT INTO items(id, tenant_id, embedding) VALUES (%s, %s, %s)"
        else:
            query = "INSERT INTO items(id, embedding) VALUES (%s, %s)"
            
        for i, embedding in enumerate(X):
            try:
                if self.IS_TENANT_AWARE:
                    cur.execute(query, (i, self.TENANT_ID, embedding.tolist()))
                else:
                    cur.execute(query, (i, embedding.tolist()))
                    
                if (i + 1) % self.BATCH_SIZE == 0:
                    conn.commit()
                    print(f"Inserted {i + 1}/{X.shape[0]} rows")
            except Exception as e:
                print(e)
                conn.rollback()
                break
        
        # Commit any remaining rows
        conn.commit()
        print(f"Inserted {X.shape[0]} rows")
        
    def _set_tenant_context(self, cursor, tenant_id) -> None:
        if self.tenant_aware:
            cursor.execute(
                sql.SQL(""" set local nile.tenant_id = {} """).format(
                    sql.Literal(tenant_id)
                )
            )

    def _create_index(self, cur):
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                """
                with s as (select set_config('statement_timeout', '30 min', true)) select * from s;
                CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)
                """ % (self._m, self._ef_construction)
            )
        elif self._metric == "euclidean":
            cur.execute("""
                        with s as (select set_config('statement_timeout', '30 min', true)) select * from s;
                        CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d)
                        """ % (self._m, self._ef_construction))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    # We assume a pre-existing database with the extension registered
    # Nile currently does not support COPY, so we will use batch insertions
    # Nile currently does not support storage parameters, so we rely on pgvector's default
    # TODO: Actual batch insertions, or at least don't commit after each row
    # TODO: add log table for tracking various test runs and execution times
    # TODO: Make setup optional
    def fit(self, X: numpy.array):
        print("connecting to database..." + self._connection_string)
        conn = psycopg.connect(self._connection_string, autocommit=True)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        self._create_table(cur, X.shape[1])
        conn.commit()
        self._insert_data(cur, conn, X)
        self._create_index(cur)
        conn.commit()
        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        self._set_tenant_context(self._cur, self.TENANT_ID)
        query = sql.SQL(self._query).format(limit=sql.Literal(n))
        self._cur.execute(query, {"query_embedding": v}, binary=False, prepare=False)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def __str__(self):
        return f"PGVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
