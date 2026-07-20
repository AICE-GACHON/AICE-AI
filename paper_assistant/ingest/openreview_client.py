"""OpenReview API v2 클라이언트 (인증 + 페이지네이션 + 백오프).

익명 /notes 요청은 ChallengeRequiredError(봇 검증)에 막히므로
반드시 계정 로그인 후 Bearer 토큰으로 요청한다.
"""
import logging
import time

import requests

from paper_assistant import config

API = "https://api2.openreview.net"
PAGE_SIZE = 1000
MAX_RETRIES = 5

log = logging.getLogger(__name__)


class OpenReviewClient:
    def __init__(self, username: str | None = None, password: str | None = None):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "paper-assistant/0.1 (academic project)"
        username = username or config.OPENREVIEW_USERNAME
        password = password or config.OPENREVIEW_PASSWORD
        if not username or not password:
            raise RuntimeError(
                "OpenReview 자격 증명이 없습니다. .env에 OPENREVIEW_USERNAME/"
                "OPENREVIEW_PASSWORD를 설정하세요 (.env.example 참고).")
        self._login(username, password)

    def _login(self, username: str, password: str) -> None:
        r = self.session.post(f"{API}/login",
                              json={"id": username, "password": password},
                              timeout=30)
        r.raise_for_status()
        token = r.json()["token"]
        self.session.headers["Authorization"] = f"Bearer {token}"
        log.info("OpenReview 로그인 성공: %s", username)

    def _get(self, path: str, **params) -> dict:
        for attempt in range(MAX_RETRIES):
            r = self.session.get(f"{API}{path}", params=params, timeout=60)
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning("rate limit, %ds 대기", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"{path}: {MAX_RETRIES}회 재시도 후에도 실패")

    def iter_notes(self, **query):
        """페이지네이션을 처리하며 note를 하나씩 yield."""
        offset = 0
        while True:
            data = self._get("/notes", **query, limit=PAGE_SIZE, offset=offset)
            notes = data.get("notes", [])
            yield from notes
            offset += len(notes)
            if len(notes) < PAGE_SIZE:
                return

    def get_venue_group(self, venue_id: str) -> dict:
        """venue 그룹 메타데이터 (submission invitation id 등 확인용)."""
        data = self._get("/groups", id=venue_id)
        groups = data.get("groups", [])
        return groups[0] if groups else {}

    def iter_submissions(self, venue_id: str):
        """해당 venue의 모든 제출 논문 (reject/withdrawn 포함)."""
        yield from self.iter_notes(invitation=f"{venue_id}/-/Submission")

    def get_forum_replies(self, forum_id: str) -> list[dict]:
        """논문 forum의 모든 리플라이 (리뷰/메타리뷰/rebuttal/decision)."""
        return list(self.iter_notes(forum=forum_id))
