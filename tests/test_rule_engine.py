"""규칙 엔진(결제금액 함수 → 구간별 최적 카드) 검증. 표준 라이브러리만 사용.

시나리오: 카페 결제. 실제 보유 카드 특성 반영.
- 토스뱅크: 정액 캐시백 (1만원 미만 100원 / 1만원 이상 500원), 전월실적 없음
- K-패스 하나: 카페 1% 캐시백, 월 5천원 한도, 전월실적 30만원 이상
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from card_rag.rules.engine import best_by_amount  # noqa: E402
from card_rag.rules.types import ClauseCalc  # noqa: E402

TOSS_LOW = ClauseCalc("tossbank-check", "t1", "카페", "청구할인", "flat",
                      flat_amount=100, flat_min_txn=0, min_spend=0, remaining_cap=5000)
TOSS_HIGH = ClauseCalc("tossbank-check", "t2", "카페", "청구할인", "flat",
                       flat_amount=500, flat_min_txn=10000, min_spend=0, remaining_cap=5000)
KPASS_CAFE = ClauseCalc("kpass-hana-check", "k1", "카페", "청구할인", "percent",
                        rate=1.0, min_spend=300000, remaining_cap=5000)


class RuleEngineTest(unittest.TestCase):
    def test_threshold_recommendation(self):
        # K-패스 실적 충족(30만) → 토스뱅크 정액과 K-패스 1%가 결제액에 따라 갈림
        segs = best_by_amount([TOSS_LOW, TOSS_HIGH, KPASS_CAFE],
                              prev_month_spend={"tossbank-check": 0, "kpass-hana-check": 300000})
        cards = [(s.a_from, s.a_to, s.card_id) for s in segs]
        self.assertEqual(cards, [
            (0, 10000, "tossbank-check"),        # 1만원 미만: 토스 100원 > K-패스(<100)
            (10000, 50000, "tossbank-check"),    # 1만~5만: 토스 500원 > K-패스(100~500)
            (50000, None, "kpass-hana-check"),   # 5만 이상: K-패스 1% > 토스 500원
        ])

    def test_min_spend_filters_out_kpass(self):
        # K-패스 실적 미충족(20만 < 30만) → K-패스 혜택절 제외 → 전 구간 토스뱅크
        segs = best_by_amount([TOSS_LOW, TOSS_HIGH, KPASS_CAFE],
                              prev_month_spend={"tossbank-check": 0, "kpass-hana-check": 200000})
        cards = [(s.a_from, s.a_to, s.card_id) for s in segs]
        self.assertEqual(cards, [
            (0, 10000, "tossbank-check"),
            (10000, None, "tossbank-check"),
        ])

    def test_no_eligible_returns_empty_segment(self):
        segs = best_by_amount([KPASS_CAFE], prev_month_spend={"kpass-hana-check": 0})
        self.assertEqual(len(segs), 1)
        self.assertIsNone(segs[0].card_id)


if __name__ == "__main__":
    unittest.main()
