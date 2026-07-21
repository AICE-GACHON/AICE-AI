"""파일럿 샘플의 실제 값을 뜯어본다 — 스키마 정규화 레이어 확정용."""
import json
from collections import Counter

from paper_assistant import config

recs = json.loads((config.RAW_DIR / "pilot_iclr2024" / "sample.json").read_text(encoding="utf-8"))


def v(field):
    """OpenReview v2의 {'value': x} 래핑을 벗긴다."""
    return field.get("value") if isinstance(field, dict) else field


print("===== 1. submission 핵심 필드 실제 값 =====")
for rec in recs[:3]:
    c = rec["submission"]["content"]
    print(f"  title    : {v(c.get('title',{}))[:60]}")
    print(f"  venue    : {v(c.get('venue',{}))}")
    print(f"  venueid  : {v(c.get('venueid',{}))}")
    print(f"  primary  : {v(c.get('primary_area',{}))}")
    print(f"  keywords : {str(v(c.get('keywords',{})))[:80]}")
    print(f"  abstract : {len(v(c.get('abstract','')) or '')} chars")
    print(f"  authorids: {str(v(c.get('authorids',{})))[:80]}")
    print()

print("===== 2. venue 값 분포 (= decision 대용 가능한가?) =====")
print(Counter(v(r["submission"]["content"].get("venue", {})) for r in recs))

print("\n===== 3. Official_Review 실제 값 (1건 전체) =====")
for rec in recs:
    for rep in rec["replies"]:
        if any(i.endswith("Official_Review") for i in rep.get("invitations", [])):
            c = rep["content"]
            for k in ("rating", "confidence", "soundness", "presentation", "contribution"):
                print(f"  {k:14}: {v(c.get(k, {}))}")
            for k in ("summary", "strengths", "weaknesses", "questions"):
                text = v(c.get(k, "")) or ""
                print(f"  {k:14}: [{len(text)} chars] {text[:200].replace(chr(10),' ')}...")
            print(f"  signatures    : {rep.get('signatures')}")
            break
    else:
        continue
    break

print("\n===== 4. weaknesses 길이 분포 (LLM 추출 비용 추정용) =====")
lengths = []
for rec in recs:
    for rep in rec["replies"]:
        if any(i.endswith("Official_Review") for i in rep.get("invitations", [])):
            t = v(rep["content"].get("weaknesses", "")) or ""
            lengths.append(len(t))
lengths.sort()
if lengths:
    print(f"  리뷰 수 {len(lengths)}건 | 최소 {lengths[0]} / 중앙 {lengths[len(lengths)//2]} "
          f"/ 최대 {lengths[-1]} chars | 평균 {sum(lengths)//len(lengths)}")

print("\n===== 5. Decision / Meta_Review 필드 =====")
for kind in ("Decision", "Meta_Review"):
    for rec in recs:
        for rep in rec["replies"]:
            if any(i.endswith(kind) for i in rep.get("invitations", [])):
                c = rep["content"]
                print(f"  {kind} 키: {sorted(c.keys())}")
                for k, val in c.items():
                    print(f"    {k}: {str(v(val))[:120]}")
                break
        else:
            continue
        break

print("\n===== 6. 논문↔리뷰 연결 방식 확인 =====")
rec = recs[1]
print(f"  submission id={rec['submission']['id']} forum={rec['submission']['forum']}")
for rep in rec["replies"][:4]:
    print(f"    reply id={rep['id']} forum={rep.get('forum')} replyto={rep.get('replyto')} "
          f"inv={[i.split('/')[-1] for i in rep.get('invitations',[])]}")
