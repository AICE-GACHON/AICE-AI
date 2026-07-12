"""triage 로직 검증. 표준 라이브러리만 사용 → 의존성 설치 없이 실행 가능.

실행: python -m unittest tests.test_triage   (레포 루트에서)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from card_rag.rag.triage import triage  # noqa: E402
from card_rag.rag.types import RAG_JUDGE, RULE_ONLY, MerchantCtx, RuleCandidate  # noqa: E402


def paths(decisions):
    return {d.card_id: d.path for d in decisions}


class TriageTest(unittest.TestCase):
    def test_no_condition_all_rule_only(self):
        m = MerchantCtx("m1", "스타벅스 강남", "카페", category_uncertain=False)
        cands = [
            RuleCandidate("A", "ca", "카페", "적립", 1000, 1000, has_condition=False),
            RuleCandidate("B", "cb", "카페", "적립", 800, 800, has_condition=False),
        ]
        self.assertEqual(paths(triage(cands, m)), {"A": RULE_ONLY, "B": RULE_ONLY})

    def test_ambiguous_but_rank_stable_is_rule_only(self):
        # A는 조건 애매하지만, 제외돼도(1500) 여전히 1등 → 순위 불변 → LLM 불필요
        m = MerchantCtx("m2", "어느 카페", "카페")
        cands = [
            RuleCandidate("A", "ca", "카페", "적립", 2000, 1500, has_condition=True),
            RuleCandidate("B", "cb", "카페", "적립", 1000, 1000, has_condition=False),
        ]
        self.assertEqual(paths(triage(cands, m)), {"A": RULE_ONLY, "B": RULE_ONLY})

    def test_ambiguous_flips_top1_is_rag_judge(self):
        # A 제외 시 500으로 떨어져 B(1000)에게 1등을 내줌 → 순위 좌우 → 판정 필요
        m = MerchantCtx("m3", "개인 카페", "카페")
        cands = [
            RuleCandidate("A", "ca", "카페", "적립", 1200, 500, has_condition=True),
            RuleCandidate("B", "cb", "카페", "적립", 1000, 1000, has_condition=False),
        ]
        self.assertEqual(paths(triage(cands, m)), {"A": RAG_JUDGE, "B": RULE_ONLY})

    def test_category_uncertain_triggers_but_and_gate_filters(self):
        # 업종 매핑이 임베딩 폴백(불확실) → A,B 모두 조건 애매.
        # 그러나 B는 자격이 뒤집혀도 값이 그대로(1000/1000)라 순위 불변 → rule_only.
        # A만 pivotal(1200 vs 0) → rag_judge.
        m = MerchantCtx("m4", "모호한 상호", "카페", category_uncertain=True)
        cands = [
            RuleCandidate("A", "ca", "카페", "적립", 1200, 0, has_condition=False),
            RuleCandidate("B", "cb", "카페", "적립", 1000, 1000, has_condition=False),
        ]
        self.assertEqual(paths(triage(cands, m)), {"A": RAG_JUDGE, "B": RULE_ONLY})


if __name__ == "__main__":
    unittest.main()
