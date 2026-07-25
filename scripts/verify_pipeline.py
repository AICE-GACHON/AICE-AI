"""LangGraph 파이프라인 배선 검증 ($0, LLM off).

DB에 이미 적재된 논문으로 analyze()를 끝까지 돌려 Report 구조를 확인한다.
크레딧을 쓰지 않으므로 (use_llm=False) 태깅·종합 요약은 스텁이지만,
검색→병렬 분석→종합의 DAG 흐름과 Report 스키마를 검증할 수 있다.
"""
import logging

from paper_assistant import analyze

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# 알려진 주제로 쿼리 (그래프 신경망 — 파일럿 데이터에 유사 논문 있음)
report = analyze(
    title="Graph Neural Networks for Molecular Property Prediction",
    abstract=("We propose a message-passing graph neural network for predicting "
              "molecular properties, evaluated on quantum chemistry benchmarks "
              "with strong baselines and ablation studies."),
    use_llm=False,   # $0 — 스텁 태깅/요약
)

print(f"\n{'='*70}")
print(f"쿼리: {report.query_title}")
print(f"{'='*70}")

print(f"\n[유사 논문 {len(report.similar_papers)}편]")
for p in report.similar_papers[:8]:
    pct = f"{p.similarity_percentile:.0f}%ile" if p.similarity_percentile else "-"
    print(f"  {p.rank:2}. [{pct:>7}] {p.decision:14} {p.title[:48]}")
    for t in p.tags:
        print(f"        · {t.kind}: {t.reason}")

print(f"\n[리뷰 지적 패턴 {len(report.review_patterns)}개]")
for pat in report.review_patterns[:5]:
    print(f"  [{pat.aspect}] {pat.paper_count}/{pat.total_papers}편: {pat.label[:60]}")

print(f"\n[게재 경향 {len(report.venue_trends)}개]")
for t in report.venue_trends:
    print(f"  {t.venue}: {t.accept_count}/{t.paper_count} accept "
          f"({t.accept_rate*100:.0f}%)")

print(f"\n[종합 요약]\n{report.summary_markdown}")

print(f"\n{'='*70}")
print("✅ 파이프라인 end-to-end 정상 (LLM off, $0)")
print(f"{'='*70}")
