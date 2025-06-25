# This module defines the Nile class, which is an implementation of the BaseANN interface
# for the Nile database (a serverless Postgres designed for SaaS).
#
# The main responsibilities of the key methods are as follows:
#
# - __init__: Initializes the Nile instance with connection details, HNSW parameters,
#             and options for tenant awareness and data insertion formats.
#
# - fit: Connects to the database and prepares it for queries. This includes:
#   - Creating a table (either shared or tenant-aware) if it doesn't exist.
#   - Creating tenants if needed, picking them up if the table already exists.
#   - Inserting the vector data in batches.
#   - Creating an HNSW index on the embeddings.
#
# - query: Executes a k-NN search. For tenant-aware configurations, it also
#          handles switching the tenant context periodically.
#
# - set_query_arguments: Configures the `ef_search` parameter for HNSW queries to
#                        control the trade-off between search speed and accuracy.
#
# Helper methods for 'fit':
# - _create_table: Dispatches table creation to tenant-aware or shared versions.
# - _create_shared_table: Creates a simple table for embeddings.
# - _create_tenant_aware_table: Creates a multi-tenant table.
# - _create_tenants: Registers tenants for multi-tenant scenarios.
# - _insert_data: Manages batch insertion of data with retry logic, supporting both
#                 shared and tenant-aware schemas.
# - _create_index: Builds the HNSW index on the vector column.

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
    
    
    def __init__(self, metric: str, connection_string: str, m: int, ef_construction: int, existing_table: bool = False, table_name: str = "items", is_tenant_aware: bool = True, insert_as_text: bool = False, num_tenants: int = 50, compute_id: str = None):
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
            compute_id (str, optional): The ID of the compute resources to use for tenants. Default is None.
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
        self._compute_id = compute_id
        self._tenant_ids = []

        if metric == "angular":
            self._query = f"SELECT id, %(query_embedding)s::vector<=>embedding as distance FROM {table_name} ORDER BY distance LIMIT {{limit}}"
        elif metric == "euclidean":
            self._query = f"SELECT id, %(query_embedding)s::vector<->embedding as distance FROM {table_name} ORDER BY distance LIMIT {{limit}}"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        
    def _create_table(self, cur, conn, dimensions):
        if self.IS_TENANT_AWARE:
            self._create_tenant_aware_table(cur, dimensions)
        else:
            self._create_shared_table(cur, dimensions)

    def _create_shared_table(self, cur, dimensions):
        print("creating table...")
        cur.execute(f"DROP INDEX IF EXISTS {self._table_name}_embedding_idx") # needed because Nile doesn't support CASCADE
        cur.execute(f"DROP TABLE IF EXISTS {self._table_name}")
        cur.execute(f"CREATE TABLE {self._table_name} (id int, embedding vector(%d))" % dimensions)
        cur.execute(f"ALTER TABLE {self._table_name} ALTER COLUMN embedding SET STORAGE PLAIN")
        
    def _create_tenant_aware_table(self, cur, dimensions):
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
        
    def _create_tenants(self, cur, conn):
        try:
            self._tenant_ids = [str(uuid.uuid4()) for _ in range(self.NUM_TENANTS)]
            tenants_to_insert = [(tenant_id, f"tenant_{i}") for i, tenant_id in enumerate(self._tenant_ids)]
            conn.commit() # DML operations on tenants table have to be first in transaction
            # Nile requires each tenant insert to be in a separate transaction
            for i, tenant_id in enumerate(self._tenant_ids):
                if self._compute_id:
                    cur.execute("INSERT INTO tenants(id, name, compute_id) VALUES (%s, %s, %s)", (tenant_id, f"tenant_{i}", self._compute_id))
                else:
                    cur.execute("INSERT INTO tenants(id, name) VALUES (%s, %s)", (tenant_id, f"tenant_{i}"))
                conn.commit()
        except Exception as e:
            print(f"Error during tenant insertion: {e}") # Log other errors if any
            # If ON CONFLICT is not supported or another error occurs, this might need specific handling
            
    def _get_existing_tenants(self, cur, conn):
        """
        Fetch the most recently created tenants from the tenants table.
        Populates self._tenant_ids with the most recent NUM_TENANTS tenant IDs.
        """
        try:
            cur.execute(
                "SELECT id FROM tenants ORDER BY created DESC LIMIT %s", (self.NUM_TENANTS,)
            )
            rows = cur.fetchall()
            self._tenant_ids = [row[0] for row in rows]
            print(f"Fetched {len(self._tenant_ids)} existing tenants.")
        except Exception as e:
            print(f"Error fetching existing tenants: {e}")

    def _insert_batch(self, cur, conn, X, columns, get_params_for_row, log_prefix=""):
        total_rows = X.shape[0]
        inserted_count = 0
        
        for i in range(0, total_rows, self.BATCH_SIZE):
            batch_indices = range(i, min(i + self.BATCH_SIZE, total_rows))
            
            retries = 3
            while retries > 0:
                try:
                    for j in range(0, len(batch_indices), self.MULTI_ROW_INSERT_SIZE):
                        chunk_indices = list(batch_indices)[j:j + self.MULTI_ROW_INSERT_SIZE]
                        if not chunk_indices:
                            continue

                        values_template = ", ".join([f"({', '.join(['%s'] * len(columns))})"] * len(chunk_indices))
                        column_names = ", ".join(columns)
                        query = sql.SQL(f"INSERT INTO {self._table_name} ({column_names}) VALUES {{}}").format(sql.SQL(values_template))
                        
                        params = []
                        for idx in chunk_indices:
                            params.extend(get_params_for_row(X[idx], idx))

                        if log_prefix:
                            print(f"{log_prefix} Inserting multi-row chunk of {len(chunk_indices)} vectors")
                        cur.execute(query, params, prepare=False)
                    
                    conn.commit()
                    inserted_count += len(batch_indices)
                    print(f"{log_prefix} Inserted {inserted_count}/{total_rows} rows")
                    break # Success, exit retry loop
                except Exception as e:
                    conn.rollback()
                    retries -= 1
                    print(f"{log_prefix} Error inserting batch: {e}")
                    if retries > 0:
                        print(f"{log_prefix} Retrying batch... ({retries} retries left)")
                    else:
                        print(f"{log_prefix} Aborting insertion due to error. {inserted_count} rows were inserted before the error.")
                        return False
        return True

    def _insert_data_tenant_aware(self, cur, conn, X):
        print(f"Loading {X.shape[0]} embeddings with {X.shape[1]} dimensions into table for {self.NUM_TENANTS} tenants")
        for tenant_id in self._tenant_ids:
            noise = numpy.random.normal(loc=0.0, scale=1e-5, size=X.shape).astype(X.dtype)
            X_tenant = X + noise
            
            def get_params(row, index):
                if self._insert_as_text:
                    vector_str = "[" + ",".join(map(str, row)) + "]"
                    return [index, tenant_id, vector_str]
                else:
                    return [index, tenant_id, row]

            print(f"Inserting {X.shape[0]} vectors for tenant {tenant_id}...")
            success = self._insert_batch(cur, conn, X_tenant, ["id", "tenant_id", "embedding"], get_params, log_prefix=f"  Tenant {tenant_id[:8]}:")
            if not success:
                print(f"Aborting insertion for tenant {tenant_id} due to error.")
                return # Stop further insertions for this tenant
    
    def _insert_data_shared(self, cur, conn, X):
        print(f"Loading {X.shape[0]} embeddings with {X.shape[1]} dimensions into table")
        
        def get_params(row, index):
            if self._insert_as_text:
                vector_str = "[" + ",".join(map(str, row)) + "]"
                return [index, vector_str]
            else:
                return [index, row]
        
        success = self._insert_batch(cur, conn, X, ["id", "embedding"], get_params)
        if not success:
            print("Aborting insertion due to error.")

    def _insert_data(self, cur, conn, X):
        print("inserting data...")
        if self.IS_TENANT_AWARE:
            self._insert_data_tenant_aware(cur, conn, X)
        else:
            self._insert_data_shared(cur, conn, X)

        print(f"Successfully finished inserting data.")

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
            self._create_tenants(cur, conn)
            conn.commit() # Commit DDL changes for table creation
            
            self._insert_data(cur, conn, X) # This method will handle its own batch commits/rollbacks
            
            self._create_index(cur)
            conn.commit() # Commit DDL changes for index creation
        else:
            self._get_existing_tenants(cur, conn)
        print("done!")
        # The connection cursor is set up. Tenant context for queries will be set within the query method.
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)
        # using strict_order to avoid modifying the query and using CTE
        self._cur.execute("SET hnsw.iterative_scan = strict_order;")

    def query(self, v, n):
        if self.IS_TENANT_AWARE:
            self._query_count += 1
            # Switch tenant context every 200 queries.
            if (self._query_count - 1) % 200 == 0:
                tenant_id_to_set = random.choice(self._tenant_ids)
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
        if self._compute_id:
            s += f", compute_id={self._compute_id}"
        s += ")"
        return s