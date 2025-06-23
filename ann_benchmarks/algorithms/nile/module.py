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
    MULTI_ROW_INSERT_SIZE = 100
    NUM_TENANTS = 50
    
    
    def __init__(self, metric: str, connection_string: str, m: int, ef_construction: int, existing_table: bool = False, table_name: str = "items", is_tenant_aware: bool = True, insert_as_text: bool = False, num_tenants: int = 50):
        """
        Initialize the Nile ANN class.

        Parameters:
            metric (str): The distance metric to use ('angular' or 'euclidean').
            connection_string (str): PostgreSQL connection string for the database.
            m (int): HNSW parameter for the number of bi-directional links created for every new element during construction.
            ef_construction (int): HNSW parameter controlling index construction accuracy/speed tradeoff.
            existing_table (bool, optional): If True, use an existing table instead of creating a new one. Default is False.
            table_name (str, optional): Name of the table to use or create. Default is 'items'.
            is_tenant_aware (bool, optional): If True, enables tenant-aware mode (multi-tenant table and logic). Default is True.
            insert_as_text (bool, optional): If True, inserts vectors as text (string) instead of binary (numpy array). Useful for debugging or compatibility issues. Default is False (binary).
            num_tenants (int, optional): Number of tenants to use in tenant-aware mode. Default is 50.
        """
        self._metric = metric
        self._connection_string = connection_string
        self._m = m
        self._ef_construction = ef_construction
        self._existing_table = existing_table
        self._ef_search = 40 # default, it can be overridden by the query arguments
        self._cur = None
        self._query_count = 0
        self._table_name = table_name
        self.IS_TENANT_AWARE = is_tenant_aware
        self._insert_as_text = insert_as_text
        self.NUM_TENANTS = num_tenants

        if self.IS_TENANT_AWARE:
            self._tenant_ids = [str(uuid.uuid4()) for _ in range(self.NUM_TENANTS)]
            self._large_tenant_id = self._tenant_ids[0]
            self._other_tenant_ids = self._tenant_ids[1:]

        if metric == "angular":
            self._query = f"SELECT id, %(query_embedding)s::vector<=>embedding as distance FROM {table_name} ORDER BY distance LIMIT {{limit}}"
        elif metric == "euclidean":
            self._query = f"SELECT id, %(query_embedding)s::vector<->embedding as distance FROM {table_name} ORDER BY distance LIMIT {{limit}}"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
    def _create_table(self, cur, conn, dimensions):
        if self.IS_TENANT_AWARE:
            self._create_tenant_aware_table(cur, conn, dimensions)
        else:
            self._create_shared_table(cur, dimensions)

    def _create_shared_table(self, cur, dimensions):
        print("creating table...")
        cur.execute(f"DROP INDEX IF EXISTS {self._table_name}_embedding_idx") # needed because Nile doesn't support CASCADE
        cur.execute(f"DROP TABLE IF EXISTS {self._table_name}")
        cur.execute(f"CREATE TABLE {self._table_name} (id int, embedding vector(%d))" % dimensions)
        cur.execute(f"ALTER TABLE {self._table_name} ALTER COLUMN embedding SET STORAGE PLAIN")
        
    def _create_tenant_aware_table(self, cur, conn,dimensions):
        print("creating table...")
        try:
            cur.execute(f"DROP INDEX IF EXISTS {self._table_name}_embedding_idx") # needed because Nile doesn't support CASCADE
        except Exception as e:
            print(e) # ignore if index doesn't exist, this is a workaround for THE-2303
            
        try:
            cur.execute(f"DROP TABLE IF EXISTS {self._table_name}")
            cur.execute(f"CREATE TABLE {self._table_name} (id int, tenant_id uuid, embedding vector(%d))" % dimensions) 
            cur.execute(f"ALTER TABLE {self._table_name} ALTER COLUMN embedding SET STORAGE PLAIN")
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

            inserted_count = 0
            # For each tenant, insert the entire dataset
            for tenant_id in self._tenant_ids:
                print(f"Inserting {total_rows} vectors for tenant {tenant_id}...")
                for i in range(0, total_rows, self.BATCH_SIZE):
                    batch_indices = range(i, min(i + self.BATCH_SIZE, total_rows))
                    
                    retries = 3
                    while retries > 0:
                        try:
                            # Process the batch in smaller chunks using multi-row INSERTs
                            for j in range(0, len(batch_indices), self.MULTI_ROW_INSERT_SIZE):
                                chunk_indices = list(batch_indices)[j:j + self.MULTI_ROW_INSERT_SIZE]
                                if not chunk_indices:
                                    continue

                                values_template = ", ".join(["(%s, %s, %s)"] * len(chunk_indices))
                                query = sql.SQL(f"INSERT INTO {self._table_name} (id, tenant_id, embedding) VALUES {{}}" ).format(sql.SQL(values_template))

                                params = []
                                for idx in chunk_indices:
                                    if self._insert_as_text:
                                        # Convert numpy array to pgvector-compatible string
                                        vector_str = "[" + ",".join(map(str, X[idx])) + "]"
                                        params.extend([idx, tenant_id, vector_str])
                                    else:
                                        params.extend([idx, tenant_id, X[idx]])

                                print(f"  Inserting multi-row chunk of {len(chunk_indices)} vectors for tenant {tenant_id}")
                                # Can't use prepared statements due to a known issue with Nile
                                cur.execute(query, params, prepare=False)

                            conn.commit()
                            inserted_count += len(batch_indices)
                            print(f"  Total inserted rows for tenant {tenant_id}: {inserted_count}/{total_rows}")
                            break # Success, exit retry loop
                        except Exception as e:
                            conn.rollback()
                            retries -= 1
                            print(f"Error inserting batch for tenant {tenant_id}: {e}")
                            if retries > 0:
                                print(f"Retrying batch... ({retries} retries left)")
                            else:
                                print(f"Aborting insertion for tenant {tenant_id} due to error. {inserted_count} rows were inserted before the error.")
                                return # Stop further insertions for this tenant
                inserted_count = 0  # Reset for next tenant
        else:
            # Non-tenant-aware insertion logic
            print(f"Loading {total_rows} embeddings with {X.shape[1]} dimensions into table")
            
            inserted_count = 0
            for i in range(0, total_rows, self.BATCH_SIZE):
                batch_start_idx = i
                batch_end_idx = min(i + self.BATCH_SIZE, total_rows)
                
                retries = 3
                while retries > 0:
                    try:
                        # Process the batch in smaller chunks using multi-row INSERTs
                        for j in range(batch_start_idx, batch_end_idx, self.MULTI_ROW_INSERT_SIZE):
                            chunk_start_idx = j
                            chunk_end_idx = min(j + self.MULTI_ROW_INSERT_SIZE, batch_end_idx)
                            num_in_chunk = chunk_end_idx - chunk_start_idx

                            if num_in_chunk == 0:
                                continue

                            values_template = ", ".join(["(%s, %s)"] * num_in_chunk)
                            query = sql.SQL(f"INSERT INTO {self._table_name} (id, embedding) VALUES {{}}" ).format(sql.SQL(values_template))
                            
                            params = []
                            for k in range(chunk_start_idx, chunk_end_idx):
                                if self._insert_as_text:
                                    vector_str = "[" + ",".join(map(str, X[k])) + "]"
                                    params.extend([k, vector_str])
                                else:
                                    params.extend([k, X[k]])
                            
                            cur.execute(query, params, prepare=False)

                        conn.commit()
                        inserted_count = batch_end_idx
                        print(f"Inserted {inserted_count}/{total_rows} rows")
                        break # Success, exit retry loop
                    except Exception as e:
                        conn.rollback()
                        retries -= 1
                        print(f"Error inserting batch starting at row {batch_start_idx}: {e}")
                        if retries > 0:
                             print(f"Retrying batch... ({retries} retries left)")
                        else:
                            print(f"Aborting insertion due to error. {inserted_count} rows were inserted before the error.")
                            return # Stop further insertions

        print(f"Successfully inserted all {total_rows} rows.")

    def _create_index(self, cur):
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                f"""
                CREATE INDEX {self._table_name}_embedding_idx ON {self._table_name} USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)
                """ % (self._m, self._ef_construction)
            )
        elif self._metric == "euclidean":
            cur.execute(
                f"""
                CREATE INDEX {self._table_name}_embedding_idx ON {self._table_name} USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d)
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
                
                self._cur.execute(sql.SQL("commit; set nile.tenant_id = {}").format(sql.Literal(tenant_id_to_set)))

        query = sql.SQL(self._query).format(limit=sql.Literal(n))
        if self._insert_as_text:
            # Convert query vector to pgvector-compatible string
            v_str = "[" + ",".join(map(str, v)) + "]"
            self._cur.execute(query, {"query_embedding": v_str}, binary=False, prepare=False)
        else:
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
