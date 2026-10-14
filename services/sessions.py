"""ความจำบทสนทนาต่อ session (cookie)

เก็บในหน่วยความจำ: รีสตาร์ตแล้วหาย และใช้ได้กับเซิร์ฟเวอร์เดียว
บน serverless (Vercel) แต่ละ request อาจไปตกคนละ instance จึงอาจ "ลืม" ได้
หน้า UI จึงส่ง history มาเองทุกครั้งด้วย (ถ้าส่ง history มา ระบบจะใช้ค่านั้นแทน)
ถ้าจะใช้จริงหลายเครื่อง ควรย้ายไป Redis / Vercel KV โดยใช้ session_id เป็น key ได้เลย
"""

import uuid
from collections import OrderedDict

from fastapi import Response

SESSION_COOKIE = "session_id"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 วัน


class SessionStore:
    def __init__(self, max_sessions: int = 1000, max_messages: int = 12):
        self.max_sessions = max_sessions
        self.max_messages = max_messages
        self._data: OrderedDict[str, list[dict]] = OrderedDict()

    def ensure(self, session_id: str | None) -> tuple[str, bool]:
        """คืน (session_id, เป็น session ใหม่หรือไม่)"""
        if session_id and session_id in self._data:
            self._data.move_to_end(session_id)
            return session_id, False
        new_id = uuid.uuid4().hex
        self._data[new_id] = []
        while len(self._data) > self.max_sessions:  # กันหน่วยความจำโตไม่จำกัด
            self._data.popitem(last=False)
        return new_id, True

    def get(self, session_id: str) -> list[dict]:
        return list(self._data.get(session_id, []))

    def append(self, session_id: str, user_text: str, reply: str) -> None:
        hist = self._data.get(session_id, []) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        self._data[session_id] = hist[-self.max_messages:]

    def reset(self, session_id: str) -> None:
        self._data[session_id] = []

    def clear_all(self) -> None:
        self._data.clear()


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax", max_age=SESSION_MAX_AGE)
