FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile
# Note: We are not running the DB in this container, it is just for testing. 
# The test script will connect to an external Nile database(cloud or containerized)

RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends
USER root
RUN python3 -m pip install psycopg[binary] pgvector