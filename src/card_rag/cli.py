"""인제스천/운영 CLI.  실행: `card-rag <command>` 또는 `python -m card_rag.cli <command>`."""
from __future__ import annotations

import json
from pathlib import Path

import typer

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


@app.command()
def load(card_id: str) -> None:
    """[6] 검수된 혜택절 임베딩 + DB 적재. 카드 메타는 data/cards.json에서 읽음."""
    from card_rag.ingestion.load import load_card

    meta = json.loads(CARDS_JSON.read_text("utf-8"))[card_id]
    n = load_card(card_id, **meta)
    typer.echo(f"✅ 적재 완료: {card_id} 혜택절 {n}개")


@app.command()
def ingest(card_id: str) -> None:
    """[2]+[3] normalize→extract 까지 실행하고 검수를 위해 멈춘다(자동 load 안 함)."""
    normalize(card_id)
    extract(card_id)
    typer.echo("⏸  검수 단계: clauses JSON 확인 후 `card-rag load <card_id>` 실행")


if __name__ == "__main__":
    app()
