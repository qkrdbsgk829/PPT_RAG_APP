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


def normalize_answer_headings(answer):
    """
    LLM이 '• 핵심 결론'처럼 일반 bullet로 제목을 출력해도
    화면에서는 Markdown 섹션 제목으로 보이도록 보정한다.
    """
    if not answer:
        return ""

    heading_map = {
        "핵심 결론": "## 1. 핵심 결론",
        "확인 내용: 내부 자료 기준": "## 2. 확인 내용: 내부자료 기준",
        "확인 내용: 내부자료 기준": "## 2. 확인 내용: 내부자료 기준",
        "확인 내용: 외부 자료 기준": "## 3. 확인 내용: 외부자료 기준",
        "확인 내용: 외부자료 기준": "## 3. 확인 내용: 외부자료 기준",
        "종합 해석": "## 4. 종합 해석",
        "사용자 프로젝트 관점의 시사점": "## 5. 사용자 프로젝트 관점의 시사점",
        "실행 과제 또는 적용 방안": "## 6. 실행 과제 또는 적용 방안",
        "실행 과제": "## 6. 실행 과제 또는 적용 방안",
        "적용 방안": "## 6. 실행 과제 또는 적용 방안",
        "한계 및 추가 확인 필요사항": "## 7. 한계 및 추가 확인 필요사항",
    }

    normalized_lines = []
    for raw_line in answer.splitlines():
        stripped = raw_line.strip()

        # 이미 Markdown 제목이면 그대로 둔다.
        if stripped.startswith("#"):
            normalized_lines.append(raw_line)
            continue

        # bullet/numbered 형태 제목을 섹션 제목으로 변환
        candidate = stripped
        for prefix in ["•", "-", "*", "·"]:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):].strip()

        # "1. 핵심 결론" 같은 형태 제거
        import re
        candidate_no_num = re.sub(r"^\d+[\.\)]\s*", "", candidate).strip()

        # 제목 뒤에 콜론만 있는 경우 처리
        candidate_key = candidate_no_num.rstrip(":：").strip()

        if candidate_key in heading_map and len(candidate_key) <= 25:
            normalized_lines.append("")
            normalized_lines.append(heading_map[candidate_key])
            normalized_lines.append("")
        else:
            normalized_lines.append(raw_line)

    return "\n".join(normalized_lines).strip()


def strip_llm_source_section(answer):
    """
    LLM 본문에 생성된 '출처' 섹션은 제거한다.
    출처/관련청크/외부링크는 app.py의 하단 커스텀 섹션에서만 한 번 보여준다.
    """
    if not answer:
        return ""

    import re
    lines = answer.splitlines()
    kept = []
    skipping = False

    source_heading_patterns = [
        r"^\s*#{1,6}\s*\d*[\.)]?\s*출처\s*$",
        r"^\s*\d+[\.)]\s*출처\s*$",
        r"^\s*[•\-*·]\s*출처\s*$",
        r"^\s*출처\s*$",
        r"^\s*\d+[\.)]\s*Sources?\s*$",
        r"^\s*#{1,6}\s*Sources?\s*$",
    ]

    def is_source_heading(line: str) -> bool:
        stripped = line.strip()
        return any(re.match(p, stripped, re.IGNORECASE) for p in source_heading_patterns)

    def is_next_major_heading(line: str) -> bool:
        stripped = line.strip()
        return bool(re.match(r"^#{1,3}\s+", stripped) or re.match(r"^\d+[\.)]\s+", stripped))

    for line in lines:
        if is_source_heading(line):
            skipping = True
            continue

        if skipping:
            # 출처 섹션은 보통 마지막이라 끝까지 제거한다.
            # 혹시 이후 새 주요 섹션이 나오면 그 섹션부터 다시 살린다.
            if is_next_major_heading(line) and not is_source_heading(line):
                skipping = False
                kept.append(line)
            continue

        kept.append(line)

    return "\n".join(kept).strip()


def first_sentence_preview(text, max_chars=180):
    """청크 내용에서 첫 문장에 가까운 미리보기를 만든다."""
    import re
    text = (text or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""

    # OCR/Markdown 잡음 완화
    text = text.replace("[원본 텍스트]", "").replace("[상세 해석 및 의미]", "")
    text = re.sub(r"#{1,6}\s*", "", text).strip()

    # 한국어/영어 문장부호 기준 첫 문장 추출
    m = re.search(r"(.{30,}?[\.\!\?。])\s", text)
    if m:
        preview = m.group(1).strip()
    else:
        preview = text[:max_chars].strip()

    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "..."
    return preview



def extract_chunk_ids_from_text(text):
    """답변/컨텍스트 문자열 안에 남아 있는 chunk_id를 추출한다."""
    import re
    text = text or ""
    patterns = [
        r"ppt\d+_chunk_\d+",
        r"PPT\d+_chunk_\d+",
        r"ppt_\d+_chunk_\d+",
    ]
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    # 원본 chunk_id는 보통 소문자이므로 비교 안정화를 위해 소문자화
    normalized = []
    for cid in found:
        cid2 = cid.strip()
        if cid2 not in normalized:
            normalized.append(cid2)
    return normalized


def enrich_result_with_chunk_fallback(result):
    """
    LangGraph 최종 state에서 retrieved_chunks가 비어 있더라도
    답변/컨텍스트에 chunk_id가 남아 있으면 app.py의 chunk map으로 복구한다.
    이 함수가 있어야 관련 청크 표시와 그래프 노란색 하이라이트가 깨지지 않는다.
    """
    if result is None:
        return {}

    result = dict(result)
    existing_chunks = result.get("retrieved_chunks", []) or []
    if existing_chunks:
        return result

    text_pool = "\n".join([
        str(result.get("answer", "") or ""),
        str(result.get("context", "") or ""),
        str(result.get("web_analysis_context", "") or ""),
    ])

    chunk_ids = extract_chunk_ids_from_text(text_pool)
    if not chunk_ids:
        return result

    chunk_map = load_all_chunks()
    restored = []
    seen = set()

    # 대소문자/경로 차이 대응용 index
    lower_index = {str(k).lower(): v for k, v in chunk_map.items()}

    for cid in chunk_ids:
        key = cid.lower()
        chunk = lower_index.get(key)
        if not chunk:
            continue

        real_cid = chunk.get("chunk_id", cid)
        if real_cid in seen:
            continue
        seen.add(real_cid)

        restored.append({
            "chunk_id": real_cid,
            "document": chunk.get("document", ""),
            "text": chunk.get("text", ""),
            "metadata": chunk.get("metadata", {}),
            "score": None,
        })

    if restored:
        result["retrieved_chunks"] = restored

    return result

def get_answer_basis(result):
    """답변이 어떤 근거를 사용했는지 UI에 표시하기 위한 값."""
    retrieved_chunks = result.get("retrieved_chunks", []) or []
    web_results = result.get("web_results", []) or []
    web_analysis = result.get("web_analysis", {}) or {}
    web_analysis_context = str(result.get("web_analysis_context", "") or "")
    intent = str(result.get("intent", "")).upper()

    # Tavily 결과가 web_results에 없더라도 web_analysis / web_analysis_context에 남는 경우가 있어 fallback 처리
    has_web_analysis_sources = bool(web_analysis.get("sources"))
    has_web_context = (
        "[외부검색]" in web_analysis_context
        and "외부검색 결과가 없습니다" not in web_analysis_context
        and "외부검색은 실행하지 않았습니다" not in web_analysis_context
        and "외부검색 미실행" not in web_analysis_context
        and "TAVILY_API_KEY가 설정되지 않아" not in web_analysis_context
    )
    used_web = bool(result.get("user_approved_web_search", False)) or bool(web_results) or has_web_analysis_sources or has_web_context

    if retrieved_chunks and used_web:
        return "내부 + 외부자료 기준"
    if retrieved_chunks:
        return "내부자료 기준"
    if used_web:
        return "외부자료 기준"
    if intent in ["REPORT", "SUMMARY", "CHAT"]:
        return "대화맥락 기반 작성"
    return "LLM 작성/정리"


def format_used_chunks(result, max_items=5):
    chunks = result.get("retrieved_chunks", []) or []
    if not chunks:
        return ""

    lines = []
    for chunk in chunks[:max_items]:
        doc = chunk.get("document", "")
        cid = chunk.get("chunk_id", "")
        preview = first_sentence_preview(chunk.get("text", ""), max_chars=180)

        lines.append(f"- **{doc} / {cid}**")
        if preview:
            lines.append(f"  - {preview}")

    return "\n".join(lines)


def format_web_sources(result, max_items=7):
    """외부 출처 표시. web_results가 비어도 web_analysis.sources를 fallback으로 사용."""
    web_results = result.get("web_results", []) or []
    web_analysis = result.get("web_analysis", {}) or {}
    analysis_sources = web_analysis.get("sources", []) or []

    lines = []

    if web_results:
        for i, item in enumerate(web_results[:max_items], start=1):
            title = item.get("title", "") or "제목 없음"
            url = item.get("url", "") or ""
            if url:
                lines.append(f"- **W{i}.** [{title}]({url})")
            else:
                lines.append(f"- **W{i}.** {title}")
        return "\n".join(lines)

    # fallback: analyze_web_results_node가 만든 sources 사용
    for i, item in enumerate(analysis_sources[:max_items], start=1):
        source_id = item.get("source_id", f"W{i}")
        title = item.get("title", "") or "제목 없음"
        url = item.get("url", "") or ""
        if url:
            lines.append(f"- **{source_id}.** [{title}]({url})")
        else:
            lines.append(f"- **{source_id}.** {title}")

    return "\n".join(lines)


def build_assistant_message(result):
    """
    사용자에게 보여줄 최종 답변만 반환한다.
    intent, retrieval_grade 같은 내부 디버그 값은 화면에 노출하지 않는다.
    근거 구분은 답변 아래의 '근거 확인' 박스에서만 보여준다.
    """
    answer = strip_llm_source_section(normalize_answer_headings(clean_answer_text(result.get("answer", ""))))

    if not answer.strip():
        answer = "답변 생성 결과가 비어 있습니다. 질문을 조금 더 구체화해서 다시 시도해 주세요."

    return answer


def build_conversation_context(max_turns=6):
    """
    현재 화면에 표시된 최근 대화 이력을 rag_core.py에 전달하기 위한 문자열로 만든다.
    "이것/저것/아까/방금" 같은 지시어 해소에 사용된다.
    """
    qa_pairs = st.session_state.get("qa_pairs", [])
    if not qa_pairs:
        return "현재 화면 대화 이력 없음"

    recent_pairs = qa_pairs[-max_turns:]
    blocks = []

    for i, qa in enumerate(recent_pairs, start=1):
        question = qa.get("question", "")
        answer = qa.get("answer", "")

        # 너무 긴 답변은 오케스트레이션 단계에서 부담이 되므로 일부만 전달
        answer_preview = answer[:2500]

        blocks.append(f"""
[화면 대화 {i}]
사용자 질문:
{question}

어시스턴트 답변:
{answer_preview}
""")

    return "\n".join(blocks)


def execute_question(user_question, force_web_search=False):
    with st.spinner(
        "내외부자료 검색 및 답변 생성 중..." if force_web_search else "내외부자료 검색 및 답변 생성 중..."
    ):
        result = run_integrated_rag(
            user_question,
            force_web_search=force_web_search,
            conversation_context=build_conversation_context(),
        )

    # v7: result state에서 retrieved_chunks가 비는 경우에도
    # 답변/컨텍스트에 남은 chunk_id를 기준으로 관련 청크와 그래프 하이라이트를 복구한다.
    result = enrich_result_with_chunk_fallback(result)

    retrieved_chunks = result.get("retrieved_chunks", []) or []

    # 내부자료가 실제로 검색된 경우에만 그래프 노드를 새로 하이라이트한다.
    # 보고서 작성/요약처럼 검색을 쓰지 않는 질문에서는 직전 내부검색 하이라이트를 유지한다.
    if retrieved_chunks:
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
    align-self: flex-start;
    height: calc(100vh - 24px);
    max-height: calc(100vh - 24px);
    overflow-y: auto;
    z-index: 5;
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

.source-box {
    margin-top: 18px;
    padding: 14px 16px;
    border-radius: 14px;
    background: #F8FAFC;
    border: 1px solid #E5E7EB;
    font-size: 13px;
    color: #334155;
}
.source-box-title {
    font-weight: 900;
    margin-bottom: 6px;
    color: #0F172A;
}


/* 그래프 패널 sticky 보강: Streamlit 버전별 DOM 차이를 감안해 선택자를 여러 개 둔다 */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
div[data-testid="stHorizontalBlock"] > div:nth-child(2),
section.main div[data-testid="column"]:nth-of-type(2) {
    position: sticky !important;
    top: 12px !important;
    align-self: flex-start !important;
    height: calc(100vh - 24px) !important;
    max-height: calc(100vh - 24px) !important;
    overflow-y: auto !important;
    z-index: 10 !important;
}

/* agraph iframe에 마우스가 올라가면 휠이 그래프 zoom/pan에 잡힐 수 있어 오른쪽 패널 스크롤바를 항상 보이게 한다 */
div[data-testid="stHorizontalBlock"] > div:nth-child(2)::-webkit-scrollbar {
    width: 8px !important;
}


/* ===== v5 FINAL: 답변 제목 가시성 강화 ===== */
div[data-testid="stMarkdownContainer"] h2 {
    margin-top: 1.45rem !important;
    margin-bottom: 0.75rem !important;
    padding: 0.72rem 0.95rem !important;
    border-left: 6px solid #2563EB !important;
    background: #F1F5F9 !important;
    border-radius: 12px !important;
    font-size: 23px !important;
    font-weight: 900 !important;
    color: #0F172A !important;
    line-height: 1.35 !important;
}

div[data-testid="stMarkdownContainer"] h3 {
    margin-top: 1.15rem !important;
    margin-bottom: 0.55rem !important;
    font-size: 19px !important;
    font-weight: 850 !important;
    color: #1E293B !important;
}

/* ===== v5 FINAL: 근거 박스 가시성 강화 ===== */
.source-box {
    margin-top: 28px !important;
    padding: 18px 20px !important;
    border-radius: 18px !important;
    background: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05) !important;
}
.source-box-title {
    font-size: 19px !important;
    font-weight: 950 !important;
    color: #0F172A !important;
}

/* ===== v5 FINAL: 오른쪽 그래프 패널 sticky 안정화 ===== */
section.main > div.block-container > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
    position: sticky !important;
    top: 16px !important;
    align-self: flex-start !important;
    height: auto !important;
    max-height: none !important;
    min-height: 680px !important;
    overflow: visible !important;
    z-index: 10 !important;
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

        st.caption("Enter로 질문하고, Shift+Enter로 줄바꿈할 수 있습니다. 내부/외부/작성 모드는 자동 판단합니다.")

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

        submitted_question = selected_quick_question

        if submitted_question:
            qa = execute_question(submitted_question, force_web_search=False)
            st.session_state.qa_pairs.append(qa)
            st.rerun()

    else:
        top_c1, top_c2 = st.columns([5, 1])
        with top_c1:
            st.caption("대화 맥락을 유지합니다. '아까 답변', '이 내용', '방금 결과' 같은 표현을 사용할 수 있습니다.")
        with top_c2:
            if st.button("대화 초기화", use_container_width=True):
                st.session_state.qa_pairs = []
                st.session_state.last_result = None
                st.session_state.highlighted_node_ids = set()
                st.session_state.selected_node = None
                st.rerun()

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

            used_chunks_text = format_used_chunks(result)
            web_sources_text = format_web_sources(result)
            basis = get_answer_basis(result)
            if used_chunks_text or web_sources_text:
                st.markdown("---")
                st.markdown("## 8. 출처")
                st.markdown(f"**사용 근거:** {basis}")

                if used_chunks_text:
                    st.markdown("### 관련 청크")
                    st.markdown(used_chunks_text)

                if web_sources_text:
                    st.markdown("### 외부 출처")
                    st.markdown(web_sources_text)
            if False and grade == "bad" and not used_web:
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
# 7. Bottom Chat Input: Enter 제출 / Shift+Enter 줄바꿈
# ============================================================

chat_placeholder = (
    "무엇을 도와드릴까요? 내부/외부/작성 모드는 자동 판단합니다."
    if len(st.session_state.qa_pairs) == 0
    else "추가 질문을 입력하세요. 예: 방금 답변을 보고서 형태로 정리해줘."
)

chat_question = st.chat_input(chat_placeholder)

if chat_question and chat_question.strip():
    qa = execute_question(chat_question.strip(), force_web_search=False)
    st.session_state.qa_pairs.append(qa)
    st.rerun()
