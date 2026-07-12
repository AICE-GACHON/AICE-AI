"""condition_signal(가맹점↔조건 매칭 + 관대 폴백) 검증. 표준 라이브러리만 사용."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from card_rag.rag.retrieval import condition_signal  # noqa: E402


class ConditionSignalTest(unittest.TestCase):
    def test_no_exclude_condition_applies(self):
        s = condition_signal([1.0, 0.0], include_vec=None, exclude_vec=None)
        self.assertTrue(s.lean_included)
        self.assertTrue(s.decisive)

    def test_include_clearly_wins_is_decisive(self):
        s = condition_signal([1.0, 0.0], include_vec=[1.0, 0.0], exclude_vec=[0.0, 1.0])
        self.assertTrue(s.lean_included)
        self.assertTrue(s.decisive)

    def test_borderline_defers_to_llm(self):
        # 포함/제외 유사도 격차가 근소 → decisive=False (LLM 판정으로)
        s = condition_signal([1.0, 0.0], include_vec=[1.0, 1.0], exclude_vec=[1.0, 0.9])
        self.assertFalse(s.decisive)
        self.assertTrue(s.lean_included)  # 관대: 애매하면 적용 쪽으로 기움

    def test_exclude_only_low_similarity_is_lenient_apply(self):
        s = condition_signal([1.0, 0.0], include_vec=None, exclude_vec=[0.0, 1.0])
        self.assertTrue(s.lean_included)   # 제외에 안 걸림 → 적용(관대)
        self.assertTrue(s.decisive)

    def test_exclude_only_high_similarity_defers(self):
        s = condition_signal([1.0, 0.0], include_vec=None, exclude_vec=[1.0, 0.2])
        self.assertFalse(s.lean_included)
        self.assertFalse(s.decisive)      # 제외 신호 강함 → LLM 판정


if __name__ == "__main__":
    unittest.main()
