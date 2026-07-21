"""SPECTER2 논문 임베딩.

논문 1편 = 벡터 1개. 청킹하지 않는다 — SPECTER2가 title+abstract 수준의
paper-level 표현을 학습한 모델이라, 본문을 쪼개 넣으면 오히려 노이즈가 된다.

입력 형식은 학습 시와 동일하게 `title + [SEP] + abstract`를 지켜야 한다.
임베딩은 마지막 레이어의 CLS 토큰.

adapter 종류 (같은 base 위에 갈아끼움):
- proximity : 논문↔논문 유사도 검색 (본 프로젝트의 용도)
- adhoc_query: 짧은 키워드 쿼리 → 논문 검색
"""
import bisect
import logging

import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

BASE_MODEL = "allenai/specter2_base"
ADAPTERS = {
    "proximity": "allenai/specter2",
    "adhoc_query": "allenai/specter2_adhoc_query",
}
MAX_LENGTH = 512

log = logging.getLogger(__name__)


class Specter2Embedder:
    def __init__(self, adapter: str = "proximity", device: str | None = None):
        if adapter not in ADAPTERS:
            raise ValueError(f"알 수 없는 adapter: {adapter} (가능: {list(ADAPTERS)})")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("SPECTER2 로드 중 (adapter=%s, device=%s)...", adapter, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoAdapterModel.from_pretrained(BASE_MODEL)
        self.model.load_adapter(ADAPTERS[adapter], source="hf",
                                load_as=adapter, set_active=True)
        self.model.to(self.device).eval()
        self.dim = self.model.config.hidden_size
        log.info("로드 완료 — 임베딩 차원 %d", self.dim)

    def _format(self, title: str, abstract: str) -> str:
        return f"{title or ''}{self.tokenizer.sep_token}{abstract or ''}"

    @torch.no_grad()
    def encode(self, papers: list[tuple[str, str]], batch_size: int = 16):
        """(title, abstract) 리스트 → (N, dim) 텐서.

        반환 벡터는 L2 정규화되어 있다. pgvector에서 코사인 거리를 쓸 때
        정규화된 벡터끼리는 내적과 동일해져 계산이 단순해진다.
        """
        vectors = []
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            texts = [self._format(t, a) for t, a in batch]
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                return_tensors="pt", return_token_type_ids=False,
            ).to(self.device)
            output = self.model(**inputs)
            cls = output.last_hidden_state[:, 0, :]  # CLS 토큰
            vectors.append(torch.nn.functional.normalize(cls, p=2, dim=1).cpu())
        return torch.cat(vectors) if vectors else torch.empty(0, self.dim)

    def encode_one(self, title: str, abstract: str):
        return self.encode([(title, abstract)])[0]


# --- 코사인 유사도 → 백분위 변환 -------------------------------------------
#
# SPECTER2의 코사인 값은 0.72~0.98의 좁은 구간에 압축되어 있다.
# ICLR 2024 논문 300편(무작위 쌍 89,700개)으로 실측한 분포:
#   평균 0.845 / 표준편차 0.033 / 중앙값 0.845
# 즉 **무관한 논문쌍도 0.84가 나온다**. 원시 코사인 값을 그대로
# "유사도 84%"처럼 노출하면 사용자가 반드시 오해한다.
#
# 아래 참조 분위수로 원시 점수를 백분위로 바꿔서 노출한다.
# (scripts/measure_similarity_dist.py 로 재측정 가능)

_REFERENCE_QUANTILES = [
    (0.7708, 1.0), (0.7924, 5.0), (0.8233, 25.0), (0.8448, 50.0),
    (0.8668, 75.0), (0.8998, 95.0), (0.9231, 99.0), (0.9442, 99.9),
]
_REF_SCORES = [s for s, _ in _REFERENCE_QUANTILES]
_REF_PERCENTILES = [p for _, p in _REFERENCE_QUANTILES]


def similarity_percentile(cosine: float) -> float:
    """코사인 유사도 → 백분위(0~100). 클수록 무작위 쌍 대비 유사하다는 뜻.

    예: 0.92 → 약 99.0 (무작위 쌍의 99%보다 유사 = 상위 1%)
    선형 보간이며, 참조 구간을 벗어나면 0/100에 수렴한다.
    """
    if cosine <= _REF_SCORES[0]:
        return max(0.0, _REF_PERCENTILES[0] * cosine / _REF_SCORES[0])
    if cosine >= _REF_SCORES[-1]:
        return 100.0

    i = bisect.bisect_right(_REF_SCORES, cosine)
    s0, s1 = _REF_SCORES[i - 1], _REF_SCORES[i]
    p0, p1 = _REF_PERCENTILES[i - 1], _REF_PERCENTILES[i]
    return p0 + (p1 - p0) * (cosine - s0) / (s1 - s0)
