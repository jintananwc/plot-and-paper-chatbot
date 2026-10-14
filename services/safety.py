"""มาตรการป้องกันการใช้งานที่ไม่พึงประสงค์ (Safety) — ตรวจหลายชั้นเรียงจากถูก/เร็ว ไปแพง/ช้า

ข้อความผู้ใช้ (input)                                  คำตอบของบอท (output)
 1) Rate limit ต่อ IP + ความยาว/ไฟล์ภาพ (chat_pipeline)  1) ตรวจ prompt รั่ว (canary)
 2) Blocklist คำหยาบ (blocklist.txt)                    2) ปิดบังข้อมูลส่วนบุคคล
 3) หัวข้อต้องห้ามของร้าน (forbidden_words.txt)          3) Blocklist คำหยาบ
    จับได้แม้แทรกช่องว่าง/เครื่องหมาย/แยกตัวอักษร          4) Llama Guard 4 (direction=output)
    และมี regex จับคำที่เลี่ยงด้วยความหมาย
 4) Prompt injection / jailbreak
 5) สั่งเปลี่ยนราคา/สร้างส่วนลด/อนุมัติสิทธิ์/สมมติกฎใหม่
 6) ขอข้อมูลลูกค้าคนอื่น
 7) ปิดบังข้อมูลส่วนบุคคล (PII) ก่อนส่งโมเดล
 8) Llama Guard 4 (direction=input)
ชั้น 2–6 เป็นกฎตายตัว (ไม่เรียกโมเดล ตอบเร็ว ผลแน่นอน)
ถ้า Llama Guard เรียกไม่สำเร็จ -> fail-closed (ไม่ถือว่า safe)
"""

import logging
import os
import re
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from config import Settings
from services import categories as cat
from services.llm_client import LLMError, chat_completion
from services.prompts import PROMPT_CANARY

log = logging.getLogger("chatbot")


class GuardUnavailable(Exception):
    """เรียก Llama Guard ไม่สำเร็จ หรืออ่านผลไม่ได้"""


@dataclass
class SafetyResult:
    safe: bool
    direction: str
    layer: str = "pass"  # pass | blocklist | injection | llama_guard | canary
    categories: list[str] = field(default_factory=list)
    message: str | None = None  # ข้อความปฏิเสธ (กรณี unsafe)
    redacted: list[str] = field(default_factory=list)  # ชนิดข้อมูลส่วนบุคคลที่ถูกปิดบัง

    def to_dict(self) -> dict:
        return {
            "result": "safe" if self.safe else "unsafe",
            "direction": self.direction,
            "layer": self.layer,
            "categories": self.categories,
            "redacted": self.redacted,
        }


# ---------------------------------------------------------------------------
# 1) Rate limit (หน่วยความจำในเครื่อง เหมาะกับ instance เดียว)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


# ---------------------------------------------------------------------------
# 2–3) รายการคำต้องห้าม (blocklist.txt / forbidden_words.txt) — รีโหลดอัตโนมัติเมื่อไฟล์ถูกแก้
# ---------------------------------------------------------------------------

_ZERO_WIDTH = re.compile("[​-‏⁠﻿­]")
# ตัวเลข/สัญลักษณ์ที่มักใช้แทนตัวอักษรเพื่อเลี่ยงคำ เช่น murd3r, k1ller
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})


def normalize_for_match(text: str) -> str:
    """ทำให้ข้อความอยู่ในรูปเดียวกันก่อนเทียบคำต้องห้าม
    'ฆ า-ต ก ร ร ม' -> 'ฆาตกรรม', 'M.u.r.d.e.r' -> 'murder', 'murd3r' -> 'murder'
    (ตัวพิมพ์เล็ก, ตัดอักขระล่องหน, แปลงเลขเป็นตัวอักษร, ตัดทุกอย่างที่ไม่ใช่ตัวอักษรไทย/อังกฤษ)"""
    t = unicodedata.normalize("NFKC", text).casefold()
    t = _ZERO_WIDTH.sub("", t).translate(_LEET)
    return re.sub(r"[^a-z฀-๿]", "", t)


class TermList:
    """อ่านไฟล์รายการคำ: หนึ่งบรรทัดต่อหนึ่งคำ, '#' = คอมเมนต์, 'regex: ...' = รูปแบบ regex
    ทั้งคำและ regex เทียบกับข้อความที่ผ่าน normalize_for_match แล้ว (ไม่มีช่องว่าง/เครื่องหมาย)"""

    def __init__(self, path: str):
        self.path = Path(path)
        self._mtime: float | None = None
        self._terms: list[str] = []
        self._patterns: list[re.Pattern] = []

    def _reload_if_changed(self) -> None:
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            self._mtime, self._terms, self._patterns = None, [], []
            return
        if mtime == self._mtime:
            return
        terms, patterns = [], []
        for ln in self.path.read_text(encoding="utf-8-sig").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.lower().startswith("regex:"):
                try:
                    patterns.append(re.compile(ln[6:].strip(), re.IGNORECASE))
                except re.error as exc:
                    log.warning("regex ใน %s ไม่ถูกต้อง: %s (%s)", self.path.name, ln, exc)
            elif norm := normalize_for_match(ln):
                terms.append(norm)
        self._terms, self._patterns, self._mtime = terms, patterns, mtime
        log.info("โหลด %s: %d คำ, %d regex", self.path.name, len(terms), len(patterns))

    def matches(self, message: str) -> bool:
        self._reload_if_changed()
        text = normalize_for_match(message)
        return any(t in text for t in self._terms) or any(p.search(text) for p in self._patterns)


Blocklist = TermList  # ชื่อเดิม (ใช้ในโค้ด/เทสต์ที่อ้างถึง)


# ---------------------------------------------------------------------------
# 4–6) กฎตายตัว: prompt injection, สั่งเปลี่ยนกฎ/ราคา, ขอข้อมูลลูกค้าคนอื่น
# ---------------------------------------------------------------------------

def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# 4) พยายามล้างคำสั่ง/เปลี่ยนบทบาท/ขอดูคำสั่งภายใน
_INJECTION_RE = _compile([
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(the\s+)?(previous|prior|your)\s+(instructions?|rules?)",
    r"forget\s+(all\s+)?(the\s+)?(previous|prior|your)?\s*(instructions?|rules?)",
    r"(reveal|show|print|repeat|tell me)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules)",
    r"system\s*prompt",
    r"\bdan\s+mode\b|\bjailbreak\b|developer\s+mode",
    r"you\s+are\s+now\s+(a|an|the)\b",
    r"(ลืม|ละเลย|ไม่ต้องสนใจ|ยกเลิก|เพิกเฉย|ข้าม)\s*(คำสั่ง|กฎ|ข้อกำหนด|prompt|นโยบาย)",
    r"(เปิดเผย|บอก|แสดง|พิมพ์|ขอดู|ส่ง)\s*(คำสั่ง|prompt|พรอมต์|พร้อมต์|กฎ)\s*(ระบบ|ภายใน|ของคุณ|ที่ได้รับ|ตั้งต้น|ทั้งหมด)",
    r"(คำสั่งระบบ|พรอมต์ระบบ|พร้อมต์ระบบ|กฎที่ผู้พัฒนา)",
    r"(สวมบทบาท|แกล้งทำเป็น).*(ไม่มีข้อจำกัด|ไม่มีกฎ|ทำได้ทุกอย่าง)",
    r"โหมด\s*(นักพัฒนา|ผู้พัฒนา|ไร้ข้อจำกัด|developer)",
])

# 5) สั่งให้เปลี่ยนราคา/นโยบาย สร้างส่วนลด อนุมัติสิทธิ์ สวมบทเป็นเจ้าของร้าน หรือ "สมมติ" กฎใหม่
_OVERRIDE_RE = _compile([
    r"(เปลี่ยน|แก้|ปรับ)\s*(ค่าเช่า|ราคา|ค่าปรับ|นโยบาย|กฎ)",
    r"(สร้าง|ออก|ทำ|แจก|ขอ)\s*(คูปอง|coupon|โค้ด|voucher|บัตรกำนัล)",
    r"(create|generate|give me)\s+(a\s+)?(coupon|discount|voucher|promo)",
    r"อนุมัติ|approve",
    r"(เช่า|ยืม)\s*(ได้)?\s*ฟรี|ให้\s*ฟรี|free\s*(rent|rental|books?)|all\s+books\s+are\s+free",
    r"(ตอนนี้|ต่อไปนี้|จากนี้|ตั้งแต่นี้|ต่อจากนี้)\s*(คุณ|เธอ|แก|บอท)\s*(คือ|เป็น)",
    r"สมมติ(ว่า|ให้)?\s*(ร้าน|ทางร้าน|นโยบาย|กฎ|คุณ|เธอ|บอท)",
    r"(แล้ว|ให้|จง)\s*ตอบ(ว่า|แค่)",
    r"(pretend|assume|imagine)\s+(that\s+)?(the\s+)?(store|shop|policy|you)",
])

# 6) ขอข้อมูลของลูกค้าคนอื่น (ชื่อ/สิ่งที่เช่า/เบอร์/ที่อยู่)
_PRIVACY_RE = _compile([
    r"ลูกค้า\s*(คน|ท่าน|ราย)?\s*(อื่น|ก่อน|ก่อนหน้า|ล่าสุด|คนนั้น).{0,30}(ชื่อ|เช่า\S{0,8}อะไร|เบอร์|ที่อยู่|ข้อมูล|ประวัติ|ใคร)",
    r"(ชื่อ|เบอร์|ที่อยู่|ข้อมูล|ประวัติ).{0,15}(ลูกค้า\s*(คน|ท่าน|ราย)?\s*(อื่น|ก่อน)|ผู้เช่า|คนที่เช่า|คนที่จอง)",
    r"ใคร\s*(เป็นคน|เป็นผู้|ที่)?\s*(เช่า|ยืม|จอง)",
    r"(who|which customer)\s+(has\s+)?(rented|borrowed|reserved)",
    r"(other|previous|last)\s+customers?('s)?\s+(name|info|rental|data)",
])


def detect_injection(message: str) -> bool:
    return bool(_INJECTION_RE.search(message))


def detect_override(message: str) -> bool:
    return bool(_OVERRIDE_RE.search(message))


def detect_privacy_request(message: str) -> bool:
    return bool(_PRIVACY_RE.search(message))


# ---------------------------------------------------------------------------
# 5) ปิดบังข้อมูลส่วนบุคคล (PII) — ไม่ส่งข้อมูลอ่อนไหวไปยังผู้ให้บริการโมเดลภายนอก
# ---------------------------------------------------------------------------

_PII_RULES: list[tuple[str, re.Pattern, str]] = [
    ("thai_id", re.compile(r"(?<!\d)\d[\s-]?\d{4}[\s-]?\d{5}[\s-]?\d{2}[\s-]?\d(?!\d)"), "[เลขบัตรประชาชนถูกปิดบัง]"),
    ("card", re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)"), "[เลขบัตรถูกปิดบัง]"),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[อีเมลถูกปิดบัง]"),
    ("phone", re.compile(r"(?<!\d)0[689]\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"), "[เบอร์โทรถูกปิดบัง]"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for name, pattern, repl in _PII_RULES:
        text, n = pattern.subn(repl, text)
        if n:
            found.append(name)
    return text, found


# ---------------------------------------------------------------------------
# 6) Llama Guard 4 ผ่าน OpenRouter
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"S\d{1,2}", re.IGNORECASE)


def parse_verdict(raw: str) -> tuple[bool, list[str]]:
    """'safe' หรือ 'unsafe\\nS1,S2' -> (is_unsafe, [codes])"""
    lines = [ln.strip() for ln in (raw or "").strip().splitlines() if ln.strip()]
    if not lines:
        raise GuardUnavailable("ผลจาก Llama Guard ว่างเปล่า")
    verdict = lines[0].lower()
    if verdict == "safe":
        return False, []
    if verdict.startswith("unsafe"):
        return True, [c.upper() for c in _CODE_RE.findall(" ".join(lines))]
    raise GuardUnavailable(f"อ่านผลจาก Llama Guard ไม่ได้: {raw[:100]!r}")


def pick_blocked(codes: list[str]) -> list[str]:
    return [c for c in codes if c in cat.CATEGORIES and cat.CATEGORIES[c].blocked]


async def llama_guard(client: httpx.AsyncClient, settings: Settings, message: str, direction: str) -> list[str]:
    """คืนรายการหมวดที่ถูกบล็อก (ว่าง = safe)"""
    role = "user" if direction == "input" else "assistant"
    try:
        raw = await chat_completion(
            client, settings, [{"role": role, "content": message}],
            model=settings.guard_model, temperature=0, max_tokens=20,
        )
    except LLMError as exc:
        raise GuardUnavailable(exc.message) from exc
    is_unsafe, codes = parse_verdict(raw)
    return pick_blocked(codes) if is_unsafe else []


# ---------------------------------------------------------------------------
# รวมทุกชั้น
# ---------------------------------------------------------------------------

class SafetyLayer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.blocklist = TermList(str(settings.resolve(settings.blocklist_path)))
        self.forbidden = TermList(str(settings.resolve(settings.forbidden_words_path)))

    def check_rules(self, message: str) -> SafetyResult | None:
        """กฎตายตัวขาเข้า (ไม่เรียกโมเดล) — คืน SafetyResult ถ้าโดนบล็อก, None ถ้าผ่าน"""
        rules = [
            (self.blocklist.matches, "blocklist", cat.BLOCKLIST_CODE, cat.BLOCKLIST_MESSAGE),
            (self.forbidden.matches, "forbidden_word", cat.FORBIDDEN_CODE, cat.FORBIDDEN_MESSAGE),
            (detect_injection, "injection", cat.INJECTION_CODE, cat.INJECTION_MESSAGE),
            (detect_override, "policy_override", cat.OVERRIDE_CODE, cat.OVERRIDE_MESSAGE),
            (detect_privacy_request, "privacy", cat.PRIVACY_CODE, cat.PRIVACY_MESSAGE),
        ]
        for check, layer, code, msg in rules:
            if check(message):
                return SafetyResult(False, "input", layer, [code], msg)
        return None

    async def check_input(self, client: httpx.AsyncClient, message: str) -> tuple[SafetyResult, str]:
        """คืน (ผลตรวจ, ข้อความที่ปิดบัง PII แล้ว)"""
        if blocked := self.check_rules(message):
            return blocked, message

        clean, redacted = redact_pii(message)
        result = SafetyResult(True, "input", redacted=redacted)
        if self.settings.guard_enabled and clean.strip():
            codes = await llama_guard(client, self.settings, clean, "input")
            if codes:
                return SafetyResult(False, "input", "llama_guard", codes,
                                    cat.CATEGORIES[codes[0]].message, redacted), clean
        return result, clean

    async def check_output(self, client: httpx.AsyncClient, reply: str) -> tuple[SafetyResult, str]:
        if PROMPT_CANARY.casefold() in reply.casefold():
            return SafetyResult(False, "output", "canary", [cat.PROMPT_LEAK_CODE], cat.PROMPT_LEAK_MESSAGE), reply
        clean, redacted = redact_pii(reply)
        if self.blocklist.matches(clean):
            return SafetyResult(False, "output", "blocklist", [cat.BLOCKLIST_CODE], cat.BLOCKLIST_MESSAGE), clean
        if self.settings.guard_enabled:
            codes = await llama_guard(client, self.settings, clean, "output")
            if codes:
                return SafetyResult(False, "output", "llama_guard", codes,
                                    cat.CATEGORIES[codes[0]].message, redacted), clean
        return SafetyResult(True, "output", redacted=redacted), clean
