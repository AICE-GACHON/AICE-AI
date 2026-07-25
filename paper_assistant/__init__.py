"""paper_assistant — ML/AI 논문 리서치 어시스턴트 (AI 파트).

공개 API (백엔드 통합 계약):
    from paper_assistant import analyze
    report = analyze(title, abstract)   # -> Report (Pydantic)
"""


def analyze(*args, **kwargs):
    # 지연 import — 무거운 의존성(torch 등)을 실제 호출 시에만 로드
    from paper_assistant.graph.pipeline import analyze as _analyze
    return _analyze(*args, **kwargs)
