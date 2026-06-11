import json
import os
import glob

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from rag_core import run_integrated_rag


def load_json_file(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def node_color_by_type(node_type):
    colors = {
        "Document": "#7C3AED",
        "Topic": "#2563EB",
        "Technology": "#059669",
        "Industry": "#F97316",
        "Solution": "#DB2777",
        "Impact": "#DC2626",
        "Developer": "#0891B2",
        "User": "#4F46E5",
    }
    return colors.get(node_type, "#64748B")


def load_all_chunks():
    chunk_map = {}

    chunk_files = glob.glob("./data/PPT자료*/chunks.json")

    for file_path in chunk_files:
        chunks = load_json_file(file_path)

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

    return chunk_map


def load_all_nodes_raw():
    node_map = {}

    node_files = glob.glob("./data/PPT자료*/nodes_visual*.json")

    for file_path in node_files:
        nodes = load_json_file(file_path)

        for node in nodes:
            node_id = str(node.get("id", ""))

            if node_id and node_id not in node_map:
                node_map[node_id] = node

    return node_map


def find_nodes_by_retrieved_chunks(retrieved_chunks):
    node_map = load_all_nodes_raw()

    retrieved_chunk_ids = set()

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            retrieved_chunk_ids.add(chunk_id)

    highlighted_node_ids = set()

    for node_id, node in node_map.items():
        source_chunks = node.get("source_chunks", [])

        if any(chunk_id in retrieved_chunk_ids for chunk_id in source_chunks):
            highlighted_node_ids.add(node_id)

    return highlighted_node_ids


def build_graph_from_visual_json(highlighted_node_ids=None):
    if highlighted_node_ids is None:
        highlighted_node_ids = set()

    ppt_configs = [
        {
            "folder": "./data/PPT자료1",
            "nodes": "nodes_visual_ppt1.json",
            "rels": "relationships_visual_ppt1.json",
        },
        {
            "folder": "./data/PPT자료2",
            "nodes": "nodes_visual_ppt2.json",
            "rels": "relationships_visual_ppt2.json",
        },
        {
            "folder": "./data/PPT자료3",
            "nodes": "nodes_visual_ppt3.json",
            "rels": "relationships_visual_ppt3.json",
        },
    ]

    graph_nodes = []
    graph_edges = []

    seen_nodes = set()
    seen_edges = set()

    for cfg in ppt_configs:
        nodes_path = os.path.join(cfg["folder"], cfg["nodes"])
        rels_path = os.path.join(cfg["folder"], cfg["rels"])

        nodes_data = load_json_file(nodes_path)
        rels_data = load_json_file(rels_path)

        for n in nodes_data:
            node_id = str(n.get("id", ""))
            label = str(n.get("name", node_id))
            node_type = str(n.get("type", "Node"))
            description = str(n.get("description", ""))

            if not node_id or node_id in seen_nodes:
                continue

            seen_nodes.add(node_id)

            is_highlighted = node_id in highlighted_node_ids

            size = 42 if is_highlighted else (
                30 if node_type == "Document" else 22
            )

            color = (
                "#FACC15"
                if is_highlighted
                else node_color_by_type(node_type)
            )

            graph_nodes.append(
                Node(
                    id=node_id,
                    label=label,
                    title=f"{node_type}\n{description}",
                    size=size,
                    color=color,
                    shape="dot",
                )
            )

        for r in rels_data:
            source_id = str(r.get("source_id", ""))
            target_id = str(r.get("target_id", ""))
            relation = str(r.get("relation", ""))

            if not source_id or not target_id:
                continue

            edge_key = f"{source_id}_{relation}_{target_id}"

            if edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)

            graph_edges.append(
                Edge(
                    source=source_id,
                    target=target_id,
                    label=relation,
                )
            )

    config = Config(
        width=780,
        height=430,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F59E0B",
        collapsible=False,
    )

    return graph_nodes, graph_edges, config


def show_selected_node_chunks(selected_node_id):
    node_map = load_all_nodes_raw()
    chunk_map = load_all_chunks()

    node = node_map.get(selected_node_id)

    if not node:
        st.info("선택한 노드 정보를 찾지 못했습니다.")
        return

    st.markdown("### 선택한 노드")
    st.write("노드명:", node.get("name", ""))
    st.write("타입:", node.get("type", ""))
    st.write("설명:", node.get("description", ""))

    source_chunks = node.get("source_chunks", [])
    source_chunks = list(dict.fromkeys(source_chunks))

    if not source_chunks:
        st.info("이 노드에 연결된 청크가 없습니다.")
        return

    st.markdown("### 연결 청크")

    for chunk_id in source_chunks[:10]:
        chunk = chunk_map.get(chunk_id)

        if not chunk:
            continue

        title = f'{chunk_id} / {chunk.get("document", "")}'

        with st.expander(title):
            st.markdown(chunk.get("text", ""))


st.set_page_config(
    page_title="PPT 기반 AI Agent",
    layout="wide",
)

st.title("PPT 기반 AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

if "highlighted_node_ids" not in st.session_state:
    st.session_state.highlighted_node_ids = set()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


left, right = st.columns([1.1, 1])


with left:
    st.subheader("AI Agent 질의")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("질문을 입력하세요")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("내부 자료 검색 및 답변 생성 중..."):
                result = run_integrated_rag(question)

                grade = result.get("retrieval_grade", "")

                if grade in ["bad", "weak"]:
                    st.session_state.pending_question = question
                else:
                    st.session_state.pending_question = None

                answer = result.get("answer", "")

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

        st.session_state.last_result = result

        retrieved_chunks = result.get("retrieved_chunks", [])
        st.session_state.highlighted_node_ids = find_nodes_by_retrieved_chunks(
            retrieved_chunks
        )


with right:
    st.subheader("검색 / 판단 결과")

    st.markdown("### 지식 그래프")

    try:
        graph_nodes, graph_edges, graph_config = build_graph_from_visual_json(
            highlighted_node_ids=st.session_state.highlighted_node_ids
        )

        if graph_nodes:
            selected_node = agraph(
                nodes=graph_nodes,
                edges=graph_edges,
                config=graph_config,
            )

            if selected_node:
                st.session_state.selected_node = selected_node
                st.success(f"선택한 노드: {selected_node}")

            if st.session_state.selected_node:
                show_selected_node_chunks(st.session_state.selected_node)

        else:
            st.info("그래프 노드가 없습니다. data 폴더의 JSON 파일을 확인하세요.")

    except Exception as e:
        st.error(f"그래프 로드 오류: {e}")

    result = st.session_state.last_result

    if (
        result is not None
        and result.get("retrieval_grade") in ["bad", "weak"]
        and st.session_state.pending_question
    ):
        st.warning("내부 자료만으로 충분한 답변이 어렵습니다.")

        if st.button("외부검색 실행"):
            with st.spinner("외부검색 기반 답변 생성 중..."):
                result = run_integrated_rag(
                    st.session_state.pending_question,
                    force_web_search=True,
                )

            st.session_state.last_result = result
            st.session_state.pending_question = None

            answer = result.get("answer", "")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

            st.rerun()

    if result is None:
        st.info("질문을 입력하면 검색 결과가 표시됩니다.")

    else:
        st.markdown("### 질문 필수 조건")
        st.write(result.get("required_conditions", []))

        st.markdown("### 내부 자료 검색 품질")
        st.write("등급:", result.get("retrieval_grade", ""))
        st.write("이유:", result.get("grade_reason", ""))
        st.write("충족 조건:", result.get("satisfied_conditions", []))
        st.write("부족 조건:", result.get("missing_conditions", []))

        st.markdown("### 외부검색")
        st.text(result.get("web_analysis_context", ""))

        # ============================================================
        # 검색된 내부 문서
        # ============================================================

        if result.get("user_approved_web_search", False):
            st.info(
                "외부검색 기반 답변입니다. "
                "내부 검색 결과는 참고용이므로 숨김 처리했습니다."
            )

        else:
            st.markdown("### 검색된 내부 문서")

            retrieved_chunks = result.get("retrieved_chunks", [])

            if retrieved_chunks:
                for chunk in retrieved_chunks:
                    title = (
                        f'{chunk.get("document", "")} / '
                        f'{chunk.get("chunk_id", "")} / '
                        f'score {chunk.get("score", 0):.4f}'
                    )

                    with st.expander(title):
                        st.markdown(chunk.get("text", ""))

            else:
                st.write("검색된 내부 문서가 없습니다.")