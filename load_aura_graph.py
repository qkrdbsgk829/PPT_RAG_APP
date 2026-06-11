import os
import json
import glob
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


def load_json(path):
    if not os.path.exists(path):
        print(f"파일 없음: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_db(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def create_node(tx, node):
    node_id = str(node.get("id", ""))
    name = str(node.get("name", node_id))
    node_type = str(node.get("type", "Node"))
    description = str(node.get("description", ""))
    source_chunks = node.get("source_chunks", [])

    tx.run(
        """
        MERGE (n:GraphNode {id: $id})
        SET n.name = $name,
            n.type = $type,
            n.description = $description,
            n.source_chunks = $source_chunks
        """,
        id=node_id,
        name=name,
        type=node_type,
        description=description,
        source_chunks=source_chunks,
    )


def create_relationship(tx, rel):
    source_id = str(rel.get("source_id", ""))
    target_id = str(rel.get("target_id", ""))
    source_name = str(rel.get("source", source_id))
    target_name = str(rel.get("target", target_id))
    source_type = str(rel.get("source_type", "Node"))
    target_type = str(rel.get("target_type", "Node"))
    relation = str(rel.get("relation", "RELATED_TO"))
    evidence_list = rel.get("evidence_list", [])
    source_chunks = rel.get("source_chunks", [])

    if not source_id or not target_id:
        return

    tx.run(
        """
        MERGE (a:GraphNode {id: $source_id})
        SET a.name = coalesce(a.name, $source_name),
            a.type = coalesce(a.type, $source_type)

        MERGE (b:GraphNode {id: $target_id})
        SET b.name = coalesce(b.name, $target_name),
            b.type = coalesce(b.type, $target_type)

        MERGE (a)-[r:RELATED_TO {relation: $relation}]->(b)
        SET r.evidence_list = $evidence_list,
            r.source_chunks = $source_chunks
        """,
        source_id=source_id,
        target_id=target_id,
        source_name=source_name,
        target_name=target_name,
        source_type=source_type,
        target_type=target_type,
        relation=relation,
        evidence_list=evidence_list,
        source_chunks=source_chunks,
    )

def main():
    node_files = glob.glob("./data/PPT자료*/nodes_visual*.json")
    rel_files = glob.glob("./data/PPT자료*/relationships_visual*.json")

    print("노드 파일:", node_files)
    print("관계 파일:", rel_files)

    with driver.session() as session:
        session.execute_write(clear_db)

        node_count = 0
        rel_count = 0

        for file_path in node_files:
            nodes = load_json(file_path)
            for node in nodes:
                session.execute_write(create_node, node)
                node_count += 1

        for file_path in rel_files:
            rels = load_json(file_path)
            for rel in rels:
                session.execute_write(create_relationship, rel)
                rel_count += 1

        result = session.run("MATCH (n) RETURN count(n) AS count")
        aura_node_count = result.single()["count"]

        result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
        aura_rel_count = result.single()["count"]

    driver.close()

    print("적재 완료")
    print("JSON 기준 노드 수:", node_count)
    print("JSON 기준 관계 수:", rel_count)
    print("AuraDB 노드 수:", aura_node_count)
    print("AuraDB 관계 수:", aura_rel_count)


if __name__ == "__main__":
    main()