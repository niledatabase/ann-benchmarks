import pgvector.psycopg
import psycopg
import numpy
import psutil
from typing import Optional
from ..base.module import BaseANN


class Nile(BaseANN):
    
    BATCH_SIZE = 1000
    
    def __init__(self, metric: str, connection_string: str, m: int, ef_construction: int):
        self._metric = metric
        self._connection_string = connection_string
        self._m = m
        self._ef_construction = ef_construction
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    # We assume a pre-existing database with the extension registered
    # Nile currently does not support COPY, so we will use batch insertions
    # Nile currently does not support storage parameters, so we rely on pgvector's default
    # TODO: add log table for tracking various test runs and execution times
    def fit(self, X: numpy.array):
        print("connecting to database..." + self._connection_string)
        conn = psycopg.connect(self._connection_string)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        conn.commit()
        print("inserting data...")
        batches: list[numpy.array] = None
        if X.shape[0] < self.BATCH_SIZE:
            batches = [X]
        else:
            splits = [x for x in range(0, X.shape[0], self.BATCH_SIZE)][1:]
            batches = numpy.split(X, splits)
        print(f"Loading {X.shape[0]} embeddings into table using {len(batches)} batches")
        for i in range(0, len(batches)):
            batch = batches[i]
            query = "INSERT INTO items(id, embedding) VALUES %%s";
            try:
                cur.executemany(query, batch)
                print(f"Inserted {i + len(batch)} rows")
                conn.commit() # commit after each batch
            except Exception as e:
                print(e)
                conn.rollback()
                break
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction)
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def __str__(self):
        return f"PGVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
