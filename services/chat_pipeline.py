"""ไปป์ไลน์ของแชท (ใช้ร่วมกันทั้ง /chat แบบ JSON และ /chat/stream แบบสตรีม)

ตรวจข้อมูลเข้า -> Safety ขาเข้า -> วิเคราะห์ภาพ -> RAG -> LLM -> Safety ขาออก
แต่ละขั้นตอนคืน dict แบบ "blocked" เมื่อต้องหยุด หรือ None เมื่อผ่าน
"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx
from fastapi import Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from config import settings
from services import categories as cat
from services.errors import ApiError, new_request_id
from services.llm_client import LLMError, chat_completion
from services.prompts import load_prompt
from services.catalog import CatalogResult
from services.rag import SearchHit
from services.safety import GuardUnavailable, SafetyResult, redact_pii
from services.sessions import SESSION_COOKIE
from services.state import catalog, kb, limiter, safety, sessions

log = logging.getLogger("chatbot")

ALLOWED_IMAGE_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Timer:
    """จับเวลาแต่ละขั้นตอน (มิลลิวินาที) ส่งกลับใน timings_ms"""

    def __init__(self):
        self.t0 = self.last = time.perf_counter()
        self.marks: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round((now - self.last) * 1000, 1)
        self.last = now

    def result(self) -> dict:
        return {**self.marks, "total": round((time.perf_counter() - self.t0) * 1000, 1)}


@dataclass
class ChatContext:
    client: httpx.AsyncClient
    request_id: str
    session_id: str
    new_session: bool
    message: str
    history: list[dict]
    image_bytes: bytes | None = None
    image_mime: str = ""
    timer: Timer = field(default_factory=Timer)
    clean_message: str = ""
    in_result: SafetyResult | None = None
    out_result: SafetyResult | None = None
    image_analysis: dict | None = None
    hits: list[SearchHit] = field(default_factory=list)
    books: CatalogResult = field(default_factory=CatalogResult)
    image_books: list = field(default_factory=list)  # หนังสือในร้านที่ตรงกับชื่อที่อ่านได้จากรูป

    def blocked(self, result: SafetyResult) -> dict:
        safety_info = {"input": self.in_result.to_dict() if self.in_result else None, "output": None}
        safety_info[result.direction] = result.to_dict()
        return {
            "request_id": self.request_id, "type": "blocked", "reply": result.message,
            "sources": [], "image_analysis": self.image_analysis,
            "safety": safety_info, "timings_ms": self.timer.result(),
        }

    def sources(self) -> list[dict]:
        """แหล่งอ้างอิงที่ใช้ตอบ: หัวข้อนโยบายที่เกี่ยวข้อง + หนังสือจากฐานข้อมูล"""
        out = [{k: v for k, v in h.to_dict().items() if k != "text"} for h in self.hits]
        out += [{"id": b.book_id, "doc": "books.csv", "title": b.title,
                 "heading": f"{b.category} · {b.status} · ค่าเช่า {b.rental_price} บาท"} for b in self.books.books]
        return out

    def answer(self, reply: str) -> dict:
        return {
            "request_id": self.request_id, "type": "answer", "reply": reply,
            "sources": self.sources(), "image_analysis": self.image_analysis,
            "safety": {"input": self.in_result.to_dict() if self.in_result else None,
                       "output": self.out_result.to_dict() if self.out_result else None},
            "timings_ms": self.timer.result(),
        }

    def remember(self, reply: str) -> None:
        sessions.append(self.session_id, self.clean_message or "(ส่งรูปภาพ)", reply)


# ---------------------------------------------------------------------------
# รับและตรวจข้อมูลเข้า
# ---------------------------------------------------------------------------

def _client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def _parse_history(raw) -> list[dict]:
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
        parsed = [HistoryItem(**i) for i in items]
    except Exception as exc:
        raise ApiError(422, "invalid_history",
                       "รูปแบบ history ไม่ถูกต้อง ต้องเป็น [{\"role\": \"user\"|\"assistant\", \"content\": \"...\"}]") from exc
    return [{"role": h.role, "content": h.content} for h in parsed]


def _clean_history(history: list[dict]) -> list[dict]:
    out = []
    for h in history[-settings.max_history_turns * 2:]:
        text, _ = redact_pii(str(h.get("content", ""))[: settings.max_message_chars * 2])
        out.append({"role": h["role"], "content": text})
    return out


def validate_image(data: bytes, mime: str, filename: str) -> str:
    """ตรวจชนิด/ขนาดภาพ คืน mime ที่ถูกต้องจาก magic bytes (กันไฟล์ปลอมนามสกุล)"""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in (filename or "") else ""
    mime = (mime or "").lower()
    if mime not in ALLOWED_IMAGE_MIME and ext not in ALLOWED_IMAGE_EXT:
        raise ApiError(415, "unsupported_image", "รองรับเฉพาะไฟล์ภาพ .jpg .png และ .webp เท่านั้น")
    if not data:
        raise ApiError(400, "empty_image", "ไฟล์ภาพว่างเปล่าหรืออ่านไม่ได้")
    if len(data) > settings.max_image_mb * 1024 * 1024:
        raise ApiError(413, "image_too_large", f"ไฟล์ภาพใหญ่เกิน {settings.max_image_mb:g} MB กรุณาย่อขนาดแล้วลองใหม่")
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ApiError(415, "unsupported_image", "ไฟล์ที่แนบมาไม่ใช่ภาพที่ถูกต้อง")


async def read_chat_request(request: Request) -> ChatContext:
    """รับได้ทั้ง application/json (ภาพเป็น base64) และ multipart/form-data (ไฟล์ภาพ)
    ตรวจความถูกต้องทั้งหมดที่นี่ — error ส่วนนี้ตอบเป็น JSON ตามปกติแม้จะเรียกแบบสตรีม"""
    if not limiter.allow(_client_key(request)):
        raise ApiError(429, "rate_limited", "ส่งข้อความถี่เกินไป กรุณารอสักครู่แล้วลองใหม่")

    raw_history, image_bytes, image_mime, image_name = None, None, "", ""
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except Exception as exc:
            raise ApiError(400, "invalid_json", "รูปแบบ JSON ไม่ถูกต้อง") from exc
        if not isinstance(body, dict):
            raise ApiError(400, "invalid_json", "body ต้องเป็น JSON object")
        message = str(body.get("message") or "")
        raw_history = body.get("history")
        image_b64 = body.get("image_base64")
        if image_b64:
            if image_b64.startswith("data:") and "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(image_b64, validate=True)
            except Exception as exc:
                raise ApiError(400, "invalid_image", "image_base64 ไม่ใช่ base64 ที่ถูกต้อง") from exc
            image_mime, image_name = str(body.get("image_mime") or "image/jpeg"), "image"
    else:
        form = await request.form()
        message = str(form.get("message") or "")
        raw_history = form.get("history")
        image = form.get("image")
        if isinstance(image, UploadFile) and image.filename:
            image_bytes, image_mime, image_name = await image.read(), image.content_type or "", image.filename

    message = message.strip()
    if not message and image_bytes is None:
        raise ApiError(400, "empty_message", "กรุณาพิมพ์ข้อความหรือแนบรูปภาพ")
    if len(message) > settings.max_message_chars:
        raise ApiError(413, "message_too_long", f"ข้อความยาวเกิน {settings.max_message_chars} ตัวอักษร กรุณาสรุปให้สั้นลง")
    if image_bytes is not None:
        image_mime = validate_image(image_bytes, image_mime, image_name)

    session_id, new_session = sessions.ensure(request.cookies.get(SESSION_COOKIE))
    # ส่ง history มาเอง (แม้เป็น []) -> ใช้ค่านั้น | ไม่ส่งมา -> ใช้ความจำของ session
    history = _parse_history(raw_history) if raw_history not in (None, "") else sessions.get(session_id)

    return ChatContext(
        client=request.app.state.http, request_id=new_request_id(),
        session_id=session_id, new_session=new_session, message=message,
        history=_clean_history(history), image_bytes=image_bytes, image_mime=image_mime,
    )


# ---------------------------------------------------------------------------
# ขั้นตอนของไปป์ไลน์
# ---------------------------------------------------------------------------

def _parse_json_object(raw: str) -> dict | None:
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    for candidate in (text, match.group(0) if match else None):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


async def step_guard_input(ctx: ChatContext) -> dict | None:
    if ctx.message:
        ctx.in_result, ctx.clean_message = await safety.check_input(ctx.client, ctx.message)
    ctx.timer.lap("guard_input")
    if ctx.in_result and not ctx.in_result.safe:
        log.info("[%s] blocked input layer=%s %s", ctx.request_id, ctx.in_result.layer, ctx.in_result.categories)
        return ctx.blocked(ctx.in_result)
    return None


async def step_vision(ctx: ChatContext) -> dict | None:
    data_url = f"data:{ctx.image_mime};base64,{base64.b64encode(ctx.image_bytes).decode('ascii')}"
    messages = [
        {"role": "system", "content": load_prompt("vision_system.txt")},
        {"role": "user", "content": [
            {"type": "text", "text": ctx.clean_message or "ช่วยดูภาพนี้ให้หน่อย"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    raw = await chat_completion(ctx.client, settings, messages, model=settings.effective_vision_model,
                                temperature=0, max_tokens=500)
    parsed = _parse_json_object(raw) or {  # โมเดลไม่ตอบ JSON -> ใช้ข้อความดิบเป็นคำบรรยาย
        "safe": True, "relevant": True, "description": raw[:800], "search_query": raw[:200]}
    ctx.image_analysis = {
        "safe": parsed.get("safe", True) is not False,
        "relevant": parsed.get("relevant", True) is not False,
        "description": str(parsed.get("description") or "").strip(),
        "detected_items": parsed.get("detected_items") or [],
        "text_in_image": str(parsed.get("text_in_image") or "").strip(),
        "search_query": str(parsed.get("search_query") or "").strip(),
    }
    ctx.timer.lap("vision")
    if not ctx.image_analysis["safe"]:
        return ctx.blocked(SafetyResult(False, "input", "vision", [cat.IMAGE_UNSAFE_CODE], cat.IMAGE_UNSAFE_MESSAGE))
    return None


def step_retrieve(ctx: ChatContext) -> None:
    """RAG: ค้นนโยบาย/ข้อมูลร้าน (knowledge/*.md) + ค้นฐานข้อมูลหนังสือ (books.csv)"""
    ia = ctx.image_analysis or {}
    query = " ".join(filter(None, [ctx.clean_message, ia.get("search_query"), ia.get("text_in_image")]))
    hits = kb.search(query, settings.rag_top_k, settings.rag_min_coverage) if query else []
    books = catalog.search(query, settings.max_books_in_context) if query else CatalogResult()

    # คำถามต่อเนื่อง เช่น "แล้วค่าเช่าล่ะ" -> ค้นรวมกับคำถามก่อนหน้า
    prev_user = next((h["content"] for h in reversed(ctx.history) if h["role"] == "user"), "")
    if prev_user and not (books.books or books.facts):
        books = catalog.search(f"{prev_user} {query}", settings.max_books_in_context)
    if prev_user and not hits:
        hits = kb.search(f"{prev_user} {query}", settings.rag_top_k, settings.rag_min_coverage)
    # หนังสือในรูป: จับคู่ชื่อที่อ่านได้จากรูปกับฐานข้อมูลแยกต่างหาก แล้วใส่ไว้หน้าสุด
    if ctx.image_analysis:
        image_text = " ".join(filter(None, [ia.get("text_in_image"), ia.get("search_query")]))
        ctx.image_books, _ = catalog.match_titles(image_text) if image_text else ([], "")
        books.books = ctx.image_books + [b for b in books.books if b not in ctx.image_books]
    ctx.hits, ctx.books = hits, books
    ctx.timer.lap("retrieval")


def build_messages(ctx: ChatContext) -> list[dict]:
    if settings.rag_policy_mode == "all":
        policy_chunks = kb.chunks  # คลังนโยบายเล็ก -> ส่งทั้งหมด กันโมเดลตอบตกหล่น
    else:
        policy_chunks = [h.chunk for h in ctx.hits]
    policy_text = "\n\n".join(c.as_context() for c in policy_chunks) or "(ไม่พบข้อมูลที่เกี่ยวข้อง)"
 
    # รายชื่อหนังสือทั้งหมด -> บอทรู้เสมอว่าร้านมี/ไม่มีเล่มไหน แม้รอบนี้จะค้นเจอแค่บางเล่ม
    book_text = (catalog.overview() +
                 "\n\nรายชื่อหนังสือทั้งหมดในร้าน (ใช้ตรวจว่าร้านมีหรือไม่มีเล่มใด — "
                 "ถ้าชื่อที่ลูกค้าถามไม่อยู่ในรายชื่อนี้ แปลว่าร้านไม่มี):\n" + catalog.title_index())
    book_ctx = ctx.books.context()
    book_text += "\n\n" + (book_ctx or "รอบนี้ไม่มีหนังสือที่ตรงกับคำถามเป็นพิเศษ — ถ้าคำถามเป็นเรื่องนโยบาย/ข้อมูลร้าน "
                                       "ให้ตอบจาก Knowledge Base ตามปกติ (ห้ามแต่งชื่อหนังสือที่ไม่อยู่ในรายชื่อ)")
 
    messages = [
        {"role": "system", "content": load_prompt("system.txt")},
        {"role": "system", "content": "ข้อมูลร้านและนโยบาย (Knowledge Base):\n\n" + policy_text},
        {"role": "system", "content": "ข้อมูลจากฐานข้อมูลหนังสือของร้าน:\n\n" + book_text},
    ]
    messages.extend(ctx.history)
    messages.append({"role": "user", "content": _user_message(ctx)})
    return messages
 
 
def _user_message(ctx: ChatContext) -> str:
    """ข้อความลูกค้ารอบนี้ — ถ้าแนบรูป ให้ผูกผลอ่านรูปไว้กับคำถามโดยตรง
    (คำว่า "เล่มนี้" ต้องหมายถึงหนังสือในรูป ไม่ใช่หนังสือที่คุยกันในข้อความก่อนหน้า)"""
    text = ctx.clean_message or "(ลูกค้าส่งรูปภาพมาโดยไม่มีข้อความ)"
    ia = ctx.image_analysis
    if not ia:
        return text
    if ctx.image_books:
        match = "ตรงกับหนังสือในร้าน: " + "; ".join(
            f"[{b.book_id}] {b.title} (สถานะ {b.status}, ค่าเช่า {b.rental_price} บาท)" for b in ctx.image_books)
    elif ia.get("text_in_image"):
        match = "ชื่อที่อ่านได้จากรูปไม่ตรงกับหนังสือเล่มใดในฐานข้อมูลของร้าน"
    else:
        match = "ระบบอ่านชื่อหนังสือจากรูปไม่ได้"
    return (
        "[ลูกค้าแนบรูปมากับข้อความนี้ — คำว่า \"เล่มนี้ / ในรูป / แบบนี้\" หมายถึงสิ่งที่อยู่ในรูปนี้ "
        "ไม่ใช่หนังสือที่คุยกันก่อนหน้า]\n"
        f"- ผลอ่านรูป: {ia.get('description') or '-'}\n"
        f"- ข้อความในรูป: {ia.get('text_in_image') or '-'}\n"
        f"- {match}\n"
        f"- เกี่ยวข้องกับร้านหรือไม่: {'เกี่ยวข้อง' if ia.get('relevant', True) else 'ไม่เกี่ยวข้อง'}\n\n"
        f"ข้อความลูกค้า: {text}"
    )


async def step_guard_output(ctx: ChatContext, reply: str) -> tuple[dict | None, str]:
    ctx.out_result, reply = await safety.check_output(ctx.client, reply)
    ctx.timer.lap("guard_output")
    if not ctx.out_result.safe:
        log.warning("[%s] blocked output layer=%s %s", ctx.request_id, ctx.out_result.layer, ctx.out_result.categories)
        return ctx.blocked(ctx.out_result), reply
    return None, reply


def error_from_exc(exc: Exception) -> tuple[int, str, str]:
    """แปลง exception ที่คาดไว้เป็น (HTTP status, code, ข้อความไทย)"""
    if isinstance(exc, GuardUnavailable):
        return 503, "guard_unavailable", "ระบบตรวจความปลอดภัยไม่พร้อมใช้งานชั่วคราว กรุณาลองใหม่อีกครั้ง"
    if isinstance(exc, LLMError):
        return (500 if exc.code == "llm_not_configured" else 502), exc.code, exc.message
    if isinstance(exc, FileNotFoundError):
        return 500, "prompt_missing", str(exc)
    raise exc


EXPECTED_ERRORS = (GuardUnavailable, LLMError, FileNotFoundError)
