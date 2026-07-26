"""PDF draft → 제목/초록 추출.

휴리스틱($0)을 기본으로 하고, llm이 주어지면 Haiku로 정제한다.
논문 PDF는 대개 첫 페이지에 제목 → 저자 → Abstract → Introduction 순서라
이 구조를 이용한다.
"""
import io
import logging
import re

log = logging.getLogger(__name__)

_ABSTRACT = re.compile(r"\babstract\b", re.IGNORECASE)
_INTRO = re.compile(r"\b(?:1\s*\.?\s*)?introduction\b", re.IGNORECASE)


def _first_pages_text(pdf_bytes: bytes, pages: int = 2) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for page in reader.pages[:pages]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _heuristic(text: str) -> tuple[str, str]:
    """첫 페이지 텍스트에서 제목/초록을 규칙 기반으로 뽑는다."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # --- 제목: Abstract 이전의 첫 의미있는 줄(들) ---
    title_lines = []
    for ln in lines:
        if _ABSTRACT.search(ln):
            break
        # 저자/이메일/소속으로 보이면 중단
        if "@" in ln or re.search(r"\buniversity\b|\binstitute\b", ln, re.I):
            break
        if len(ln) >= 6:
            title_lines.append(ln)
        if len(" ".join(title_lines)) > 180:
            break
    title = " ".join(title_lines[:3]).strip()

    # --- 초록: "Abstract"와 "Introduction" 사이 ---
    m_abs = _ABSTRACT.search(text)
    abstract = ""
    if m_abs:
        rest = text[m_abs.end():]
        m_intro = _INTRO.search(rest)
        abstract = (rest[:m_intro.start()] if m_intro else rest[:2000])
        abstract = re.sub(r"\s+", " ", abstract).strip(" :.\n")

    return title, abstract


def extract_title_abstract(pdf_bytes: bytes, llm=None) -> tuple[str, str]:
    """PDF 바이트 → (title, abstract).

    llm이 주어지면 첫 페이지 텍스트를 Haiku로 정제해 정확도를 높인다.
    """
    text = _first_pages_text(pdf_bytes)
    title, abstract = _heuristic(text)

    if llm is not None:
        from paper_assistant.graph.llm import HAIKU
        system = (
            "Extract the paper title and abstract from the first-page text of an "
            "academic PDF. Return JSON {\"title\": ..., \"abstract\": ...}. "
            "Keep the abstract verbatim; do not summarize.")
        out = llm.json(HAIKU, system, text[:4000], max_tokens=1200)
        if out.get("title"):
            title = out["title"]
        if out.get("abstract"):
            abstract = out["abstract"]

    return title.strip(), abstract.strip()
