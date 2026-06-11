import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print("URI =", uri)

driver = GraphDatabase.driver(
    uri,
    auth=(user, password)
)

with driver.session() as session:
    node_count = session.run(
        "MATCH (n) RETURN count(n) AS c"
    ).single()["c"]

    rel_count = session.run(
        "MATCH ()-[r]->() RETURN count(r) AS c"
    ).single()["c"]

    print("노드 수 =", node_count)
    print("관계 수 =", rel_count)

driver.close()