"""[2] 정규화: 카드별 raw(HTML/PDF/txt) → 깨끗한 텍스트 1개(data/normalized/{card_id}.md).

프로토타입 규모(3~4장)에서는 수집(collect)을 수동으로 하고, 여기서부터 자동화한다.
"""
from __future__ import annotations

from pathlib import Path

RAW_DIR = Path("data/raw")
NORMALIZED_DIR = Path("data/normalized")


def pdf_to_text(path: Path) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def normalize_card(card_id: str) -> Path:
    """data/raw/{card_id}/ 아래의 모든 소스를 텍스트로 변환·병합."""
    src_dir = RAW_DIR / card_id
    if not src_dir.exists():
        raise FileNotFoundError(f"원문 폴더가 없습니다: {src_dir} (약관 원문을 먼저 수집해 넣어주세요)")

    chunks: list[str] = []
    for f in sorted(src_dir.iterdir()):
        if f.suffix.lower() == ".pdf":
            chunks.append(f"<!-- source: {f.name} -->\n" + pdf_to_text(f))
        elif f.suffix.lower() in {".html", ".htm"}:
            chunks.append(f"<!-- source: {f.name} -->\n" + html_to_text(f.read_text(encoding="utf-8")))
        elif f.suffix.lower() in {".txt", ".md"}:
            chunks.append(f"<!-- source: {f.name} -->\n" + f.read_text(encoding="utf-8"))

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    out = NORMALIZED_DIR / f"{card_id}.md"
    out.write_text("\n\n".join(chunks), encoding="utf-8")
    return out
