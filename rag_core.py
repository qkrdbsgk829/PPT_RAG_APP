# ============================================================
# PPT1 + PPT2 + PPT3 기본 RAG 최종 코드
# 입력:
# - ./PPT자료1/chunks.json
# - ./PPT자료2/chunks.json
# - ./PPT자료3/chunks.json
#
# 기능:
# - 3개 PPT 청크 통합
# - OpenAI 임베딩 생성/저장
# - 질문 임베딩
# - Cosine Similarity 기반 Top-K 검색
# - GPT 답변 생성
# ============================================================

# !pip install openai numpy pandas tqdm

import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 0. 설정
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY
)

PPT_FOLDERS = {
    "PPT 1": "./PPT자료1",
    "PPT 2": "./PPT자료2",
    "PPT 3": "./PPT자료3",
}

CHUNK_FILE_NAME = "chunks.json"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

EMBEDDING_SAVE_PATH = "./ppt_1_2_3_embeddings.json"


# ============================================================
# 1. 유틸
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def get_chunk_text(chunk):
    return (
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("chunk_text")
        or chunk.get("page_content")
        or ""
    )


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# 2. PPT1,2,3 chunks.json 통합
# ============================================================

def load_all_chunks():
    all_chunks = []

    for document_name, folder_path in PPT_FOLDERS.items():
        chunk_path = os.path.join(folder_path, CHUNK_FILE_NAME)

        if not os.path.exists(chunk_path):
            print(f"[경고] chunks.json 없음: {chunk_path}")
            continue

        chunks = load_json(chunk_path)

        for i, chunk in enumerate(chunks):
            text = get_chunk_text(chunk)

            if not text.strip():
                continue

            chunk_id = chunk.get(
                "chunk_id",
                f"{document_name.replace(' ', '').lower()}_chunk_{i:04d}"
            )

            all_chunks.append({
                "chunk_id": chunk_id,
                "document": document_name,
                "text": text,
                "metadata": chunk.get("metadata", {})
            })

        print(f"{document_name} 청크 로드 완료: {len(chunks)}개")

    print(f"\n전체 사용 청크 수: {len(all_chunks)}개")
    return all_chunks


all_chunks = load_all_chunks()


# ============================================================
# 3. 임베딩 생성
# ============================================================

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def build_or_load_embeddings(all_chunks):
    # 1. 기존 임베딩 파일이 있으면 먼저 로드
    if os.path.exists(EMBEDDING_SAVE_PATH):
        print("\n기존 임베딩 파일 로드")
        embedded_chunks = load_json(EMBEDDING_SAVE_PATH)

        print(f"임베딩 청크 수: {len(embedded_chunks)}개")

        # 기존 임베딩이 정상적으로 있으면 그대로 사용
        if len(embedded_chunks) > 0:
            return embedded_chunks

        # 기존 파일은 있는데 비어 있으면 새로 생성
        print("기존 임베딩 파일이 비어있습니다. 새 임베딩을 생성합니다.")

    else:
        print("\n임베딩 파일 없음. 새 임베딩 생성 시작")

    # 2. all_chunks가 비어 있으면 중단
    if not all_chunks:
        raise ValueError("all_chunks가 비어 있습니다. chunks.json 로드 경로를 확인하세요.")

    # 3. 새 임베딩 생성
    embedded_chunks = []

    for chunk in tqdm(all_chunks):
        text = chunk.get("text", "")

        if not text.strip():
            continue

        embedding = get_embedding(text)

        embedded_chunks.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "document": chunk.get("document", ""),
            "text": text,
            "metadata": chunk.get("metadata", {}),
            "embedding": embedding
        })

    # 4. 저장
    save_json(embedded_chunks, EMBEDDING_SAVE_PATH)

    print(f"\n임베딩 저장 완료: {EMBEDDING_SAVE_PATH}")
    print(f"임베딩 생성 청크 수: {len(embedded_chunks)}개")

    return embedded_chunks


embedded_chunks = build_or_load_embeddings(all_chunks)

# ============================================================
# 4. 검색 함수
# ============================================================

def retrieve_chunks(query, top_k=5):
    query_embedding = get_embedding(query)

    results = []

    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])

        results.append({
            "score": score,
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {})
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


# ============================================================
# 5. 답변 생성 함수
# ============================================================

def build_context(retrieved_chunks):
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"""
[근거 {i}]
문서: {chunk["document"]}
청크 ID: {chunk["chunk_id"]}
유사도: {chunk["score"]:.4f}

내용:
{chunk["text"]}
""")

    return "\n".join(context_blocks)


def generate_answer(query, top_k=5):
    retrieved_chunks = retrieve_chunks(query, top_k=top_k)

    context = build_context(retrieved_chunks)

    system_prompt = """
너는 PPT 1, PPT 2, PPT 3 자료를 기반으로 답변하는 RAG Assistant다.

답변 규칙:
1. 반드시 제공된 [검색 근거] 안의 내용만 사용한다.
2. 근거에 없는 내용은 추측하지 말고 "PPT 자료 내에서 확인되지 않습니다"라고 답한다.
3. 답변은 한국어로 작성한다.
4. 컨설팅 보고서 스타일로 간결하고 구조화해서 답변한다.
5. 답변 마지막에 사용한 근거 문서와 청크 ID를 표시한다.
"""

    user_prompt = f"""
[사용자 질문]
{query}

[검색 근거]
{context}

위 검색 근거만 사용해서 답변해줘.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.choices[0].message.content

    return answer, retrieved_chunks




# ============================================================
# LangGraph 통합 최종 코드 v3
# Memory + Corrective RAG + HITL + Tavily Web Search
# + LLM 기반 Required Conditions / Relevance Gate
# + LLM 기반 Web Search Plan
#
# 전제:
# - 기본 RAG 코드가 먼저 실행되어 있어야 함
# - retrieve_chunks(question, top_k=5)
# - client
# - CHAT_MODEL
# 위 3개가 이미 정의되어 있어야 함
# ============================================================

# !pip install langgraph tavily-python

import os
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient


# ============================================================
# 0. 설정
# ============================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MEMORY_PATH = "./conversation_memory.json"
MAX_MEMORY_TURNS = 6

if "여기에" in TAVILY_API_KEY:
    raise ValueError("Tavily API Key를 실제 키로 입력하세요.")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# ============================================================
# 1. 공통 유틸
# ============================================================

def safe_json_loads(raw: str, fallback: dict):
    try:
        raw = raw.strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)

    except Exception:
        return fallback


# ============================================================
# 2. Memory 함수
# ============================================================

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return []

    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def add_memory(question, answer):
    memory = load_memory()

    memory.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer
    })

    save_memory(memory)


def format_memory(memory):
    if not memory:
        return "이전 대화 없음"

    recent = memory[-MAX_MEMORY_TURNS:]

    blocks = []

    for i, item in enumerate(recent, start=1):
        blocks.append(f"""
[이전 대화 {i}]
질문: {item["question"]}
답변: {item["answer"]}
""")

    return "\n".join(blocks)


# ============================================================
# 3. State 정의
# ============================================================

class IntegratedRAGState(TypedDict):
    question: str
    memory: List[Dict[str, Any]]
    memory_context: str

    required_conditions: List[str]
    main_intent: str

    search_question: str
    retrieved_chunks: List[Dict[str, Any]]
    context: str

    retrieval_grade: str
    grade_reason: str
    satisfied_conditions: List[str]
    missing_conditions: List[str]
    corrected: bool

    need_web_search: bool
    user_approved_web_search: bool

    web_search_plan: Dict[str, Any]
    web_results: List[Dict[str, Any]]
    web_context: str
    web_analysis: Dict[str, Any]
    web_analysis_context: str

    answer: str

# ============================================================
# 4. PPT Context Formatting
# ============================================================

def format_chunks_for_integrated_rag(retrieved_chunks):
    blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(f"""
[문서 근거 {i}]
문서: {chunk.get("document", "")}
청크 ID: {chunk.get("chunk_id", "")}
유사도: {chunk.get("score", 0):.4f}

내용:
{chunk.get("text", "")[:1800]}
""")

    return "\n".join(blocks)


# ============================================================
# 5. Load Memory Node
# ============================================================

def load_memory_node(state: IntegratedRAGState):
    memory = load_memory()

    return {
        "memory": memory,
        "memory_context": format_memory(memory)
    }


# ============================================================
# 6. Required Conditions 추출 Node
# ============================================================

def extract_required_conditions_node(state: IntegratedRAGState):
    question = state["question"]

    system_prompt = """
너는 사용자의 질문에서 RAG 검색 시 반드시 만족해야 하는 조건을 추출하는 전문가다.

목적:
사용자 질문에 답하기 위해 내부 문서가 반드시 포함해야 하는 핵심 조건을 뽑는다.

조건 예시:
- 산업: 바이오, 조선업, 자동차, 반도체, 배터리
- 기업: HL만도, 삼성전자, 현대차
- 기술: 디지털 스레드, RAG, LangGraph, Agentic AI
- 업무영역: 품질, 생산, 개발, 구매, SCM
- 산출물 유형: 사례, 주요 공정, 로드맵, 아키텍처, 개선방안
- 특정 대상: NCR, PPAP, MIP, Claim, LLC

규칙:
1. 질문에 명시된 조건만 추출한다.
2. 질문에 없는 조건을 추론해서 만들지 않는다.
3. 너무 일반적인 표현은 제외한다. 예: "알려줘", "설명", "내용"
4. 단, 질문의 핵심 기술/산업/기업/업무영역은 반드시 포함한다.
5. JSON 형식으로만 답한다.

출력 형식:
{
  "required_conditions": ["조건1", "조건2"],
  "main_intent": "사용자가 알고 싶은 핵심 의도"
}
"""

    user_prompt = f"""
[사용자 질문]
{question}

위 질문에서 내부자료 검색 시 반드시 만족해야 하는 조건을 추출해줘.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content

    parsed = safe_json_loads(
        raw,
        fallback={
            "required_conditions": [],
            "main_intent": question
        }
    )

    return {
        "required_conditions": parsed.get("required_conditions", []),
        "main_intent": parsed.get("main_intent", question)
    }


# ============================================================
# 7. Retrieve Node
# ============================================================

def retrieve_node(state: IntegratedRAGState):
    search_question = state.get("search_question") or state["question"]

    retrieved_chunks = retrieve_chunks(
        search_question,
        top_k=5
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "context": format_chunks_for_integrated_rag(retrieved_chunks)
    }


# ============================================================
# 8. Retrieval Grader Node
# ============================================================

def grade_retrieval_node(state: IntegratedRAGState):
    question = state["question"]
    search_question = state.get("search_question") or question
    context = state.get("context", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    required_conditions = state.get("required_conditions", [])
    main_intent = state.get("main_intent", question)

    if not retrieved_chunks or not context.strip():
        return {
            "retrieval_grade": "bad",
            "grade_reason": "내부 자료에서 검색된 청크가 없습니다.",
            "satisfied_conditions": [],
            "missing_conditions": required_conditions,
            "need_web_search": True
        }

    system_prompt = """
너는 매우 엄격한 RAG 검색 결과 평가자다.

목적:
검색된 내부 자료 청크가 사용자 질문의 필수 조건을 만족하는지 판단한다.

평가 기준:
- good:
  검색된 내부 자료 근거가 required_conditions를 명시적으로 충분히 만족하고,
  main_intent에 직접 답변 가능하다.

- weak:
  일부 조건은 맞지만 핵심 조건이 부족하거나,
  일반론 수준으로만 답변 가능하다.

- bad:
  required_conditions의 핵심 조건이 검색 근거에 없다.
  단순히 일부 공통 키워드만 유사하다.

중요 규칙:
1. 검색 유사도 점수가 높아도 필수 조건이 없으면 bad 또는 weak다.
2. 질문에 명시된 산업/기업/기술/업무영역이 근거에 없으면 good을 주면 안 된다.
3. 내부 자료 근거에 없는 내용을 추론해서 있다고 판단하지 마라.
4. 단순히 '디지털 스레드', 'AI', '공정', '품질' 같은 공통어만 겹치면 bad다.
5. 자료 기반 훑어보기가 목적이므로 보수적으로 판단한다.
6. JSON 형식으로만 답한다.

출력 형식:
{
  "grade": "good" 또는 "weak" 또는 "bad",
  "reason": "판단 이유",
  "satisfied_conditions": ["충족된 조건"],
  "missing_conditions": ["부족한 조건"]
}
"""

    user_prompt = f"""
[사용자 질문]
{question}

[질문에서 추출한 필수 조건]
{json.dumps(required_conditions, ensure_ascii=False)}

[사용자 핵심 의도]
{main_intent}

[현재 검색 질문]
{search_question}

[검색된 내부 자료 근거]
{context}

위 검색 결과가 사용자 질문의 필수 조건을 만족하는지 평가해줘.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content

    parsed = safe_json_loads(
        raw,
        fallback={
            "grade": "weak",
            "reason": raw,
            "satisfied_conditions": [],
            "missing_conditions": required_conditions
        }
    )

    grade = parsed.get("grade", "weak")
    reason = parsed.get("reason", "")
    satisfied_conditions = parsed.get("satisfied_conditions", [])
    missing_conditions = parsed.get("missing_conditions", [])

    if grade not in ["good", "weak", "bad"]:
        grade = "weak"

    if missing_conditions and grade == "good":
        grade = "weak"

    return {
        "retrieval_grade": grade,
        "grade_reason": reason,
        "satisfied_conditions": satisfied_conditions,
        "missing_conditions": missing_conditions,
        "need_web_search": grade in ["weak", "bad"]
    }


# ============================================================
# 9. Corrective RAG: 내부 자료 검색용 질문 재작성
# ============================================================

def rewrite_question_node(state: IntegratedRAGState):
    original_question = state["question"]
    memory_context = state["memory_context"]
    required_conditions = state.get("required_conditions", [])
    main_intent = state.get("main_intent", original_question)
    grade_reason = state["grade_reason"]

    system_prompt = """
너는 내부 자료 RAG 검색용 질문 재작성 전문가다.

규칙:
1. 원래 질문의 의도를 유지한다.
2. required_conditions를 최대한 보존한다.
3. 이전 대화 맥락이 있으면 참고한다.
4. 내부 자료에서 검색될 법한 핵심 키워드 중심으로 바꾼다.
5. 설명 없이 검색 질문만 출력한다.
6. 한국어로 작성한다.
"""

    user_prompt = f"""
[원래 질문]
{original_question}

[필수 조건]
{json.dumps(required_conditions, ensure_ascii=False)}

[핵심 의도]
{main_intent}

[이전 대화 기억]
{memory_context}

[검색 결과가 부족했던 이유]
{grade_reason}

위 질문을 내부 자료 검색에 적합하게 다시 작성해줘.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {
        "search_question": response.choices[0].message.content.strip(),
        "corrected": True
    }


# ============================================================
# 10. HITL Node
# ============================================================

def ask_web_permission_node(state: IntegratedRAGState):
    # Streamlit의 HITL 버튼에서 force_web_search=True로 들어온 경우에만 외부검색 승인
    return {
        "user_approved_web_search": state.get("user_approved_web_search", False)
    }

# ============================================================
# 11-A. OpenAI LLM 기반 웹 검색 전략 생성
# ============================================================

def plan_web_search_with_llm(question: str):
    system_prompt = """
너는 웹 검색 전략 수립 전문가다.

목표:
사용자 질문을 보고 Tavily 검색에 적합한 검색 전략을 만든다.

규칙:
1. 특정 도메인이나 키워드를 하드코딩하지 않는다.
2. 질문의 핵심 대상, 날짜, 수치 필요 여부, 최신성 필요 여부를 판단한다.
3. 검색어는 2~3개 생성한다.
4. 한국어 검색어와 영어 검색어를 섞어도 된다.
5. 공식 사이트, 통계 사이트, 금융 사이트, 기업 사이트, 뉴스, 논문, 보고서 등 필요한 출처 유형을 판단한다.
6. 질문에 날짜가 있으면 검색어에 반드시 포함한다.
7. 실시간/현재/오늘/최신/주가/지수/환율/시세처럼 최신성이 필요한 질문은 needs_freshness=true로 둔다.
8. 숫자, 날짜, 지수, 가격, 비율처럼 정확한 값이 필요한 질문은 needs_exact_value=true로 둔다.
9. JSON 형식으로만 답한다.

출력 형식:
{
  "search_intent": "검색 목적",
  "needs_freshness": true 또는 false,
  "needs_exact_value": true 또는 false,
  "preferred_source_types": ["선호 출처 유형"],
  "queries": [
    "검색어 1",
    "검색어 2",
    "검색어 3"
  ]
}
"""

    user_prompt = f"""
[사용자 질문]
{question}

위 질문에 대한 웹 검색 전략을 만들어줘.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content

    return safe_json_loads(
        raw,
        fallback={
            "search_intent": question,
            "needs_freshness": True,
            "needs_exact_value": False,
            "preferred_source_types": [],
            "queries": [question]
        }
    )


# ============================================================
# 11. Tavily Web Search Node
# ============================================================

def web_search_node(state: IntegratedRAGState):
    question = state["question"]

    search_plan = plan_web_search_with_llm(question)

    print("\n================ 웹 검색 전략 ================\n")
    print(json.dumps(search_plan, ensure_ascii=False, indent=2))

    queries = search_plan.get("queries", [])

    if not queries:
        queries = [question]

    all_results = []
    web_blocks = []

    web_blocks.append(f"""
[외부검색 전략]
검색 목적: {search_plan.get("search_intent", question)}
최신성 필요: {search_plan.get("needs_freshness")}
정확한 수치 필요: {search_plan.get("needs_exact_value")}
선호 출처 유형: {", ".join(search_plan.get("preferred_source_types", []))}
검색어:
{chr(10).join(["- " + q for q in queries])}
""")

    for query in queries:
        print("\n[Tavily 검색어]", query)

        try:
            search_response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
                include_raw_content=True,
                auto_parameters=True
            )

            print("[Tavily results 개수]", len(search_response.get("results", [])))

            if search_response.get("answer"):
                web_blocks.append(f"""
[외부검색 요약]
검색어: {query}
요약:
{search_response.get("answer")}
""")

            results = search_response.get("results", [])

            for item in results:
                item["query"] = query
                all_results.append(item)

        except Exception as e:
            web_blocks.append(f"""
[외부검색 오류]
검색어: {query}
오류: {str(e)}
""")

    unique_results = []
    seen_urls = set()

    for item in all_results:
        url = item.get("url", "")

        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(item)

    if not unique_results:
        return {
            "web_search_plan": search_plan,
            "web_results": [],
            "web_context": f"""
[외부검색]
Tavily 검색 결과가 없습니다.

검색 전략:
{json.dumps(search_plan, ensure_ascii=False, indent=2)}
"""
        }

    for i, item in enumerate(unique_results[:10], start=1):
        web_blocks.append(f"""
[외부검색 결과 {i}]
출처번호: W{i}
검색어: {item.get("query", "")}
제목: {item.get("title", "")}
발행일: {item.get("published_date", "")}
URL: {item.get("url", "")}
내용:
{item.get("content", item.get("raw_content", ""))[:3000]}
""")

    return {
        "web_search_plan": search_plan,
        "web_results": unique_results[:10],
        "web_context": "\n".join(web_blocks)
    }


# ============================================================
# 12. Web Result Analyzer Node
# ============================================================

def analyze_web_results_node(state: IntegratedRAGState):
    question = state["question"]
    web_context = state.get("web_context", "")
    web_results = state.get("web_results", [])
    web_search_plan = state.get("web_search_plan", {})

    source_list = []

    for i, item in enumerate(web_results, start=1):
        source_list.append({
            "source_id": f"W{i}",
            "title": item.get("title", ""),
            "url": item.get("url", "")
        })

    if not web_context.strip() or not web_results:
        return {
            "web_analysis": {
                "answerable": False,
                "key_facts": [],
                "sources": source_list,
                "missing_info": "외부검색 결과 없음",
                "suggested_query": question
            },
            "web_analysis_context": f"""
[외부검색]
외부검색 결과가 없습니다.

검색 전략:
{json.dumps(web_search_plan, ensure_ascii=False, indent=2)}
"""
        }

    system_prompt = """
너는 외부검색 결과 분석 전문가다.

역할:
외부검색 결과에서 사용자의 질문에 직접 답할 수 있는 핵심 사실만 추출한다.

규칙:
1. 검색 결과에 실제로 포함된 내용만 핵심 사실로 추출한다.
2. 출처가 불분명하거나 검색 결과에 없는 내용은 만들지 않는다.
3. 각 핵심 사실에는 반드시 근거 출처번호를 붙인다.
4. 출처번호는 W1, W2, W3 형식으로 표시한다.
5. 답변 가능하면 answerable=true, 부족하면 answerable=false로 판단한다.
6. 질문이 최신 수치/현재값/오늘자 정보를 요구하는데 검색 결과가 날짜나 값을 직접 포함하지 않으면 answerable=false로 판단한다.
7. JSON 형식으로만 답한다.
8. 최종 출력은 한국어로 작성한다.

형식:
{
  "answerable": true 또는 false,
  "key_facts": [
    {
      "fact": "핵심 사실",
      "source_ids": ["W1", "W2"]
    }
  ],
  "sources": [
    {
      "source_id": "W1",
      "title": "출처 제목",
      "url": "출처 URL"
    }
  ],
  "missing_info": "부족한 정보",
  "suggested_query": "부족할 경우 다시 검색할 검색어"
}
"""

    user_prompt = f"""
[사용자 질문]
{question}

[외부검색 전략]
{json.dumps(web_search_plan, ensure_ascii=False, indent=2)}

[외부검색 결과]
{web_context}

[출처 목록]
{json.dumps(source_list, ensure_ascii=False, indent=2)}

검색 결과를 분석해줘.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content

    parsed = safe_json_loads(
        raw,
        fallback={
            "answerable": False,
            "key_facts": [],
            "sources": source_list,
            "missing_info": raw,
            "suggested_query": question
        }
    )

    key_facts = parsed.get("key_facts", [])
    sources = parsed.get("sources", source_list)

    fact_lines = []

    for item in key_facts:
        if isinstance(item, dict):
            fact = item.get("fact", "")
            source_ids = ", ".join(item.get("source_ids", []))
            fact_lines.append(f"- {fact} [{source_ids}]")
        else:
            fact_lines.append(f"- {item}")

    source_lines = []

    for src in sources:
        source_lines.append(
            f"- [{src.get('source_id')}] {src.get('title')} / {src.get('url')}"
        )

    web_analysis_context = f"""
[외부검색]
답변 가능 여부: {parsed.get("answerable")}

핵심 사실:
{chr(10).join(fact_lines) if fact_lines else "- 직접 확인된 핵심 사실 없음"}

부족한 정보:
{parsed.get("missing_info", "")}

출처:
{chr(10).join(source_lines) if source_lines else "- 출처 없음"}

추가 검색어 제안:
{parsed.get("suggested_query", "")}
"""

    return {
        "web_analysis": parsed,
        "web_analysis_context": web_analysis_context
    }


# ============================================================
# 13. Generate Node
# ============================================================

def generate_node(state: IntegratedRAGState):
    question = state["question"]
    memory_context = state["memory_context"]
    context = state["context"]
    retrieval_grade = state["retrieval_grade"]
    grade_reason = state["grade_reason"]
    required_conditions = state.get("required_conditions", [])
    satisfied_conditions = state.get("satisfied_conditions", [])
    missing_conditions = state.get("missing_conditions", [])
    user_approved_web_search = state["user_approved_web_search"]
    web_analysis_context = state.get("web_analysis_context", "")

    system_prompt = """
너는 내부 자료와 외부검색 결과를 함께 활용하는 RAG Assistant다.

답변 규칙:
1. 내부 자료 근거를 1차 근거로 사용한다.
2. 단, 내부 자료 검색 품질 평가가 good이 아닐 경우 내부 자료 내용을 억지로 답변에 사용하지 않는다.
3. retrieval_grade가 weak 또는 bad이고 외부검색도 사용하지 않았다면,
   '내부 자료에서는 해당 질문에 대한 직접 근거를 찾지 못했습니다'라고 답한다.
4. 외부검색 결과가 있으면 '외부검색 기준'이라고 구분한다.
5. 외부검색 핵심 사실을 사용할 때는 문장 끝에 출처번호를 표시한다. 예: [W1], [W2]
6. 검색 결과에 없는 내용은 절대 만들지 않는다.
7. 답변은 한국어로 작성한다. 외국어 자료는 한국어로 번역해서 작성한다.
8. 컨설팅 보고서 스타일로 간결하고 구조화한다.
9. 답변 하단에 반드시 '출처' 섹션을 둔다.
10. 외부검색이 실행되지 않은 경우에는 '내부 자료 기준'으로만 답한다.
"""

    user_prompt = f"""
[현재 질문]
{question}

[이전 대화 기억]
{memory_context}

[질문 필수 조건]
{json.dumps(required_conditions, ensure_ascii=False)}

[내부 자료 검색 품질]
등급: {retrieval_grade}
이유: {grade_reason}
충족 조건: {json.dumps(satisfied_conditions, ensure_ascii=False)}
부족 조건: {json.dumps(missing_conditions, ensure_ascii=False)}

[내부 자료 근거]
{context}

[외부검색 사용 여부]
{user_approved_web_search}

[외부검색]
{web_analysis_context}

위 근거를 바탕으로 답변해줘.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }


# ============================================================
# 14. Save Memory Node
# ============================================================

def save_memory_node(state: IntegratedRAGState):
    add_memory(
        question=state["question"],
        answer=state["answer"]
    )

    return {}


# ============================================================
# 15. 조건부 분기
# ============================================================

def route_after_grade(state: IntegratedRAGState):
    grade = state["retrieval_grade"]
    corrected = state["corrected"]

    if grade == "good":
        return "generate"

    if not corrected:
        return "rewrite_question"

    return "ask_web_permission"


def route_after_permission(state: IntegratedRAGState):
    if state["user_approved_web_search"]:
        return "web_search"

    return "generate"


# ============================================================
# 외부검색 미실행 안내 Node
# ============================================================

def skip_web_search_node(state: IntegratedRAGState):
    return {
        "web_analysis_context": """
[외부검색]
내부 자료 검색 결과만으로 충분하다고 판단되어 외부검색은 실행하지 않았습니다.
""",
        "web_analysis": {
            "answerable": None,
            "key_facts": [],
            "sources": [],
            "missing_info": "내부 자료 검색 결과만으로 충분하여 외부검색 미실행",
            "suggested_query": ""
        }
    }


# ============================================================
# 16. LangGraph Workflow
# ============================================================

workflow = StateGraph(IntegratedRAGState)

workflow.add_node("load_memory", load_memory_node)
workflow.add_node("extract_required_conditions", extract_required_conditions_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_retrieval", grade_retrieval_node)
workflow.add_node("rewrite_question", rewrite_question_node)
workflow.add_node("ask_web_permission", ask_web_permission_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("analyze_web_results", analyze_web_results_node)
workflow.add_node("generate", generate_node)
workflow.add_node("save_memory", save_memory_node)
workflow.add_node("skip_web_search", skip_web_search_node)

workflow.add_edge(START, "load_memory")
workflow.add_edge("load_memory", "extract_required_conditions")
workflow.add_edge("extract_required_conditions", "retrieve")
workflow.add_edge("retrieve", "grade_retrieval")

workflow.add_conditional_edges(
    "grade_retrieval",
    route_after_grade,
    {
        "generate": "skip_web_search",
        "rewrite_question": "rewrite_question",
        "ask_web_permission": "ask_web_permission"
    }
)

workflow.add_edge("skip_web_search", "generate")
workflow.add_edge("rewrite_question", "retrieve")

workflow.add_conditional_edges(
    "ask_web_permission",
    route_after_permission,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "analyze_web_results")
workflow.add_edge("analyze_web_results", "generate")
workflow.add_edge("generate", "save_memory")
workflow.add_edge("save_memory", END)

integrated_rag_graph = workflow.compile()


# ============================================================
# 17. 실행 함수
# ============================================================

def run_integrated_rag(question, force_web_search=False):
    result = integrated_rag_graph.invoke({
        "question": question,
        "memory": [],
        "memory_context": "",

        "required_conditions": [],
        "main_intent": "",

        "search_question": question,
        "retrieved_chunks": [],
        "context": "",

        "retrieval_grade": "",
        "grade_reason": "",
        "satisfied_conditions": [],
        "missing_conditions": [],
        "corrected": False,

        "need_web_search": False,
        "user_approved_web_search": force_web_search,

        "web_search_plan": {},
        "web_results": [],
        "web_context": "",
        "web_analysis": {},
        "web_analysis_context": "",

        "answer": ""
    })

    return result
