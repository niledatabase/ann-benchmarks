import psycopg
import pgvector.psycopg
from psycopg import sql
import numpy
import psutil
from typing import Optional
import uuid
import random
from ..base.module import BaseANN

class Nile(BaseANN):
    BATCH_SIZE = 1000 # need to tune this
    IS_TENANT_AWARE = True # TODO: make this configurable
    NUM_TENANTS = 50
    
    def __init__(self, metric: str, connection_string: str, m: int, ef_construction: int, existing_table: bool = False):
        self._metric = metric
        self._connection_string = connection_string
        self._m = m
        self._ef_construction = ef_construction
        self._existing_table = existing_table
        self._ef_search = 40 # default, it can be overridden by the query arguments
        self._cur = None
        self._query_count = 0

        if self.IS_TENANT_AWARE:
            self._tenant_ids = [str(uuid.uuid4()) for _ in range(self.NUM_TENANTS)]
            self._large_tenant_id = self._tenant_ids[0]
            self._other_tenant_ids = self._tenant_ids[1:]

        if metric == "angular":
            self._query = "SELECT id, %(query_embedding)s::vector<=>embedding as distance FROM items ORDER BY distance LIMIT {limit}"
        elif metric == "euclidean":
            self._query = "SELECT id, %(query_embedding)s::vector<->embedding as distance FROM items ORDER BY distance LIMIT {limit}"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
    def _create_table(self, cur, conn, dimensions):
        if self.IS_TENANT_AWARE:
            self._create_tenant_aware_table(cur, conn, dimensions)
        else:
            self._create_shared_table(cur, dimensions)

    def _create_shared_table(self, cur, dimensions):
        print("creating table...")
        cur.execute("DROP INDEX IF EXISTS items_embedding_idx") # needed because Nile doesn't support CASCADE
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % dimensions)
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        
    def _create_tenant_aware_table(self, cur, conn,dimensions):
        print("creating table...")
        try:
            cur.execute("DROP INDEX IF EXISTS items_embedding_idx") # needed because Nile doesn't support CASCADE
        except Exception as e:
            print(e) # ignore if index doesn't exist, this is a workaround for THE-2303
            
        try:
            cur.execute("DROP TABLE IF EXISTS items")
            cur.execute("CREATE TABLE items (id int, tenant_id uuid, embedding vector(%d))" % dimensions) 
            cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        except Exception as e:
            print(e)
            raise e
            
        try:
            tenants_to_insert = [(tenant_id, f"tenant_{i}") for i, tenant_id in enumerate(self._tenant_ids)]
            conn.commit() # DML operations on tenants table have to be first in transaction
            # Nile requires each tenant insert to be in a separate transaction
            for i, tenant_id in enumerate(self._tenant_ids):
                cur.execute("INSERT INTO tenants(id, name) VALUES (%s, %s)", (tenant_id, f"tenant_{i}"))
                conn.commit()
        except Exception as e:
            print(f"Error during tenant insertion: {e}") # Log other errors if any
            # If ON CONFLICT is not supported or another error occurs, this might need specific handling

    def _insert_data(self, cur, conn, X):
        print("inserting data...")
        total_rows = X.shape[0]

        if self.IS_TENANT_AWARE:
            print(f"Loading {total_rows} embeddings with {X.shape[1]} dimensions into table for {self.NUM_TENANTS} tenants")

            # Partition data among tenants
            indices = numpy.arange(total_rows)
            numpy.random.shuffle(indices)
            
            large_tenant_count = int(total_rows * 0.3)
            
            # vector_id -> tenant_id mapping. Keys are original vector indices (0 to N-1)
            tenant_map = {}
            
            # Assign first 30% of shuffled indices to the large tenant
            for i in range(large_tenant_count):
                tenant_map[indices[i]] = self._large_tenant_id
                
            # Assign the rest of shuffled indices randomly to other tenants
            for i in range(large_tenant_count, total_rows):
                tenant_map[indices[i]] = random.choice(self._other_tenant_ids)

            query = "INSERT INTO items(id, tenant_id, embedding) VALUES (%s, %s, %s)"
        else:
            print(f"Loading {total_rows} embeddings with {X.shape[1]} dimensions into table")
            query = "INSERT INTO items(id, embedding) VALUES (%s, %s)"

        inserted_count = 0
        for batch_start_idx in range(0, total_rows, self.BATCH_SIZE):
            batch_end_idx = min(batch_start_idx + self.BATCH_SIZE, total_rows)
            
            if self.IS_TENANT_AWARE:
                current_batch_data = [(i, tenant_map[i], X[i]) for i in range(batch_start_idx, batch_end_idx)]
            else:
                current_batch_data = [(i, X[i]) for i in range(batch_start_idx, batch_end_idx)]
            
            if not current_batch_data:
                continue # Should not happen if total_rows > 0 and BATCH_SIZE > 0

            try:
                cur.executemany(query, current_batch_data)
                conn.commit() # Commit after each successful batch
                inserted_count = batch_end_idx
                print(f"Inserted {inserted_count}/{total_rows} rows")
            except Exception as e:
                print(f"Error inserting batch starting at row {batch_start_idx}: {e}")
                conn.rollback() # Rollback the failed batch
                print(f"Aborting insertion due to error. {inserted_count} rows (from previous batches) were inserted before the error.")
                return # Stop further insertions

        print(f"Successfully inserted all {total_rows} rows.")

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
    # TODO: add log table for tracking various test runs and execution times
    # TODO: Run test with multiple pgvector parameters
    # TODO: Run on different datasets, including ones that are closer to real data
    def fit(self, X: numpy.array):
        print("connecting to database..." + self._connection_string)
        # Rely on default autocommit=False for manual transaction control
        conn = psycopg.connect(self._connection_string)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        
        if not self._existing_table:    
            self._create_table(cur, conn, X.shape[1])
            conn.commit() # Commit DDL changes for table creation
            
            self._insert_data(cur, conn, X) # This method will handle its own batch commits/rollbacks
            
            self._create_index(cur)
            conn.commit() # Commit DDL changes for index creation
        print("done!")
        # The connection cursor is set up. Tenant context for queries will be set within the query method.
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        if self.IS_TENANT_AWARE:
            self._query_count += 1
            # Switch tenant context every 200 queries.
            if (self._query_count - 1) % 200 == 0:
                # The first block of queries (1-200) is for the large tenant.
                if self._query_count == 1:
                    tenant_id_to_set = self._large_tenant_id
                else: # Subsequent blocks are for randomly chosen tenants.
                    tenant_id_to_set = random.choice(self._other_tenant_ids)
                
                self._cur.execute(sql.SQL("set nile.tenant_id = {}").format(sql.Literal(tenant_id_to_set)))

        query = sql.SQL(self._query).format(limit=sql.Literal(n))
        self._cur.execute(query, {"query_embedding": v}, binary=True, prepare=False)
        return [id for id, distance in self._cur.fetchall()]

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def __str__(self):
        s = f"Nile(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search}"
        if self.IS_TENANT_AWARE:
            s += f", tenants={self.NUM_TENANTS}"
        s += ")"
        return s
