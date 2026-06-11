import json
import os
import glob
import html

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from rag_core import run_integrated_rag


# ============================================================
# 1. 공통 함수
# ============================================================

def load_json_file(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def node_color_by_type(node_type):
    colors = {
        "Document": "#082B73",
        "Topic": "#4B9BE0",
        "Technology": "#4B9BE0",
        "Industry": "#4B9BE0",
        "Solution": "#4B9BE0",
        "Impact": "#4B9BE0",
        "Developer": "#4B9BE0",
        "User": "#4B9BE0",
        "Keyword": "#4B9BE0",
        "Page": "#DCECFB",
    }
    return colors.get(node_type, "#4B9BE0")


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

            size = 46 if is_highlighted else (34 if node_type == "Document" else 18)
            color = "#FFD400" if is_highlighted else node_color_by_type(node_type)

            graph_nodes.append(
                Node(
                    id=node_id,
                    label=label if node_type == "Document" else "",
                    title=f"{node_type}\n{label}\n{description}",
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
                    label="",
                )
            )

    config = Config(
        width=720,
        height=650,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#FFD400",
        collapsible=False,
    )

    return graph_nodes, graph_edges, config


def show_selected_node_chunks(selected_node_id):
    node_map = load_all_nodes_raw()
    chunk_map = load_all_chunks()
    node = node_map.get(selected_node_id)

    if not node:
        return

    source_chunks = node.get("source_chunks", [])
    source_chunks = list(dict.fromkeys(source_chunks))

    if not source_chunks:
        return

    st.markdown("### # 관련 청크")

    for chunk_id in source_chunks[:5]:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue

        doc = chunk.get("document", "")
        st.markdown(
            f"""
            <div class="chunk-card">
                {html.escape(chunk_id)} · {html.escape(doc)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("청크내용"):
            st.markdown(chunk.get("text", ""))


def clean_answer_text(answer):
    remove_keywords = [
        "검색 등급",
        "retrieval_grade",
        "등급이 bad",
        "등급: bad",
        "내부 자료 기준 (검색 등급",
        "검색 등급:",
        "검색등급",
    ]

    cleaned_lines = []
    for line in answer.splitlines():
        if any(keyword in line for keyword in remove_keywords):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def build_assistant_message(result):
    answer = clean_answer_text(result.get("answer", ""))
    grade = result.get("retrieval_grade", "")
    used_web = result.get("user_approved_web_search", False)

    if used_web:
        notice = "🔎 외부검색 결과를 함께 반영하여 답변드립니다."
        return f"{notice}\n\n---\n\n{answer}"

    if grade in ["good", "weak"]:
        notice = "✅ 내부 지식 저장소에서 관련 정보를 확인하였으며, 해당 내용을 기반으로 답변드립니다."
        return f"{notice}\n\n---\n\n{answer}"

    # bad일 때만 각 질문 카드의 HITL 영역에서 외부검색 버튼을 보여줌
    return answer


def execute_question(user_question, force_web_search=False):
    with st.spinner(
        "외부검색 기반 답변 생성 중..." if force_web_search else "내부 자료 검색 및 답변 생성 중..."
    ):
        result = run_integrated_rag(
            user_question,
            force_web_search=force_web_search,
        )

    retrieved_chunks = result.get("retrieved_chunks", [])
    st.session_state.highlighted_node_ids = find_nodes_by_retrieved_chunks(retrieved_chunks)
    st.session_state.last_result = result

    return {
        "question": user_question,
        "answer": build_assistant_message(result),
        "result": result,
    }


# ============================================================
# 2. Streamlit 설정 / CSS
# ============================================================

st.set_page_config(
    page_title="PPT RAG",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.6rem;
    padding-left: 1.8rem;
    padding-right: 1.8rem;
    max-width: 100%;
}

.hero-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    margin-top: 34px;
    margin-bottom: 26px;
    color: #000000;
}

h1 {
    font-size: 32px !important;
    line-height: 1.35 !important;
}

h2 {
    font-size: 24px !important;
    line-height: 1.4 !important;
}

h3 {
    font-size: 20px !important;
    line-height: 1.4 !important;
}

p, li {
    font-size: 15px !important;
    line-height: 1.75 !important;
}

section.main > div.block-container > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
    position: sticky;
    top: 12px;
    height: calc(100vh - 24px);
    overflow-y: auto;
}

section.main > div.block-container > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2)::-webkit-scrollbar {
    width: 6px;
}

.quick-btn-wrap button {
    background-color: #DDECF8 !important;
    color: #111827 !important;
    border: 0 !important;
    border-radius: 28px !important;
    height: 48px !important;
    font-weight: 700 !important;
}

.chunk-card {
    background:#FFF1B8;
    border-radius:10px;
    padding:14px 18px;
    margin-bottom:10px;
    font-weight:800;
}

.qa-card {
    border: 1px solid #E5E7EB;
    border-radius: 22px;
    padding: 26px 28px;
    margin-bottom: 34px;
    background: #FFFFFF;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

.qa-index {
    font-size: 15px;
    font-weight: 900;
    color: #64748B;
    margin-bottom: 12px;
}

.user-question-card {
    background: #F1F5F9;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 24px;
    font-weight: 900;
    font-size: 18px;
    color: #111827;
}

.answer-label {
    font-size: 15px;
    font-weight: 900;
    color: #334155;
    margin-bottom: 10px;
}

.answer-wrap {
    border-top: 1px solid #E5E7EB;
    padding-top: 22px;
}

.hitl-box {
    border: 1px solid #E5E7EB;
    border-radius: 18px;
    padding: 18px 20px;
    margin-top: 22px;
    background: #FFFFFF;
    font-weight: 800;
    color: #C00000;
}

.home-input-row {
    margin-bottom: 18px;
}

button[kind="secondary"] {
    border-radius: 24px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. Session State
# ============================================================

if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

if "highlighted_node_ids" not in st.session_state:
    st.session_state.highlighted_node_ids = set()


# ============================================================
# 4. Layout
# ============================================================

left, right = st.columns([1.45, 1])


# ============================================================
# 5. Left: 초기 화면 또는 Q&A 카드형 대화 화면
# ============================================================

with left:
    has_conversation = len(st.session_state.qa_pairs) > 0

    if not has_conversation:
        st.markdown('<div class="hero-title">PPT RAG</div>', unsafe_allow_html=True)

        with st.form("hero_question_form", clear_on_submit=True):
            input_col, submit_col = st.columns([8, 1.2], vertical_alignment="center")

            with input_col:
                hero_question = st.text_input(
                    "",
                    placeholder="무엇을 도와드릴까요?",
                    label_visibility="collapsed",
                )

            with submit_col:
                hero_submit = st.form_submit_button("질문하기", use_container_width=True)

        quick_questions = [
            "삼성디스플레이 프로젝트 요약",
            "HL MANDO 프로젝트 일정",
            "AI AGENT 프로젝트 예시",
            "프로젝트 추진 로드맵",
            "주요 기술 트렌드 요약",
            "품질 AI 과제 정리",
        ]

        qcols = st.columns(3)
        selected_quick_question = None

        st.markdown('<div class="quick-btn-wrap">', unsafe_allow_html=True)
        for i, q in enumerate(quick_questions):
            with qcols[i % 3]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    selected_quick_question = q
        st.markdown('</div>', unsafe_allow_html=True)

        submitted_question = None
        if hero_submit and hero_question:
            submitted_question = hero_question
        elif selected_quick_question:
            submitted_question = selected_quick_question

        if submitted_question:
            qa = execute_question(submitted_question, force_web_search=False)
            st.session_state.qa_pairs.append(qa)
            st.rerun()

    else:
        for idx, qa in enumerate(st.session_state.qa_pairs, start=1):
            result = qa.get("result", {})
            grade = result.get("retrieval_grade", "")
            used_web = result.get("user_approved_web_search", False)
            question_text = html.escape(qa.get("question", ""))

            st.markdown(
                f"""
                <div class="qa-card">
                    <div class="qa-index">질문 {idx}</div>
                    <div class="user-question-card">{question_text}</div>
                    <div class="answer-wrap">
                        <div class="answer-label">결과 답변</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(qa.get("answer", ""))

            if grade == "bad" and not used_web:
                st.markdown(
                    """
                    <div class="hitl-box">
                        ⚠️ 내부 문서만으로는 충분한 근거를 찾지 못했습니다.<br>
                        외부검색으로 답변을 보완할 수 있습니다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("외부검색 진행", key=f"web_search_{idx}", use_container_width=True):
                    updated_qa = execute_question(qa["question"], force_web_search=True)
                    st.session_state.qa_pairs[idx - 1] = updated_qa
                    st.rerun()

            st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# 6. Right: 그래프 + 관련 청크
# ============================================================

with right:
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

            if st.session_state.selected_node:
                show_selected_node_chunks(st.session_state.selected_node)

        else:
            st.info("그래프 노드가 없습니다.")

    except Exception as e:
        st.error(f"그래프 로드 오류: {e}")


# ============================================================
# 7. Bottom Chat Input: 대화 시작 후에는 항상 하단 입력창 사용
# ============================================================

if len(st.session_state.qa_pairs) > 0:
    followup_question = st.chat_input("질문을 입력하세요")

    if followup_question:
        qa = execute_question(followup_question, force_web_search=False)
        st.session_state.qa_pairs.append(qa)
        st.rerun()
