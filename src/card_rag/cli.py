"""인제스천/운영 CLI.  실행: `card-rag <command>` 또는 `python -m card_rag.cli <command>`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

# Windows 콘솔(cp949)에서 이모지/한글 출력이 깨지지 않도록 UTF-8로 고정
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

app = typer.Typer(help="AICE 카드혜택 추천 - RAG 인제스천 파이프라인", no_args_is_help=True)

CARDS_JSON = Path("data/cards.json")  # card_id -> {name, issuer, annual_fee, highlight}


@app.command("init-db")
def init_db_cmd() -> None:
    """pgvector 확장 + 테이블 생성."""
    from card_rag.db.base import init_db

    init_db()
    typer.echo("✅ DB 초기화 완료 (extension + tables)")


@app.command()
def normalize(card_id: str) -> None:
    """[2] data/raw/{card_id}/* → data/normalized/{card_id}.md"""
    from card_rag.ingestion.normalize import normalize_card

    out = normalize_card(card_id)
    typer.echo(f"✅ 정규화 완료: {out}")


@app.command()
def extract(card_id: str) -> None:
    """[3] 약관 → 혜택절 JSON 초안(검수 필요). data/clauses/{card_id}.json"""
    from card_rag.ingestion.extract import extract_card

    result = extract_card(card_id)
    lows = [c for c in result.clauses if c.confidence == "low"]
    typer.echo(f"✅ 추출 완료: {len(result.clauses)}개 혜택절 (검수 우선 low={len(lows)}개)")
    typer.echo(f"   → data/clauses/{card_id}.json 숫자 필드를 검수한 뒤 load 하세요.")


@app.command("extract-policies")
def extract_policies_cmd(card_id: str) -> None:
    """[3-정책] 약관 → 정책 청크(전역 제외·통합한도·특약) 추출."""
    from card_rag.ingestion.extract_policies import extract_policies

    result = extract_policies(card_id)
    typer.echo(f"✅ 정책 추출: {len(result.policies)}건 → data/policies/{card_id}.json")


@app.command()
def load(card_id: str) -> None:
    """[6] 혜택절 + 정책 청크 임베딩 + DB 적재. 카드 메타는 data/cards.json에서 읽음."""
    from card_rag.ingestion.load import load_card, load_policies

    meta = json.loads(CARDS_JSON.read_text("utf-8"))[card_id]
    n = load_card(card_id, **meta)
    p = load_policies(card_id)
    typer.echo(f"✅ 적재 완료: {card_id} 혜택절 {n}개, 정책 {p}개")


@app.command()
def ingest(card_id: str, load_data: bool = typer.Option(True, "--load/--no-load")) -> None:
    """[전체 자동] normalize→extract→자동검증→load. 사람 검수 없이 파이프라인 완주."""
    from card_rag.ingestion.extract import extract_card
    from card_rag.ingestion.extract_policies import extract_policies
    from card_rag.ingestion.validate import validate_clauses

    normalize(card_id)
    result = extract_card(card_id)
    pol = extract_policies(card_id)
    typer.echo(f"ℹ 정책 청크 {len(pol.policies)}건 추출")
    errors, warnings = validate_clauses(result)
    for w in warnings:
        typer.echo(f"⚠ {w}")
    if errors:
        for e in errors:
            typer.echo(f"✖ {e}")
        typer.echo("검증 실패 → 적재 중단. 원문/추출을 확인하세요.")
        raise typer.Exit(1)
    typer.echo(f"✅ 검증 통과: 혜택절 {len(result.clauses)}건 (경고 {len(warnings)}개)")
    if load_data:
        load(card_id)


@app.command()
def recommend(merchant: str, categories: str = "카페,음식점", spend: int = 300000) -> None:
    """가맹점 추천(end-to-end): 규칙엔진+triage+LLM판정+가드레일(LangGraph). 보유카드=전체."""
    from card_rag.rag.graph import recommend_via_graph
    from card_rag.rag.recommend import render

    meta = json.loads(CARDS_JSON.read_text("utf-8"))
    card_names = {k: v["name"] for k, v in meta.items()}
    cards = list(meta)
    cats = [c.strip() for c in categories.split(",")]
    result = recommend_via_graph(merchant, cats, cards, {c: spend for c in cards})
    typer.echo(render(result, card_names))


@app.command()
def ask(question: str) -> None:
    """자연어 질문에 혜택절 근거로 답변(B. Q&A). 예: card-rag ask "편의점 만원, 어떤 카드?" """
    from card_rag.rag.qa import answer

    typer.echo(answer(question))


if __name__ == "__main__":
    app()
