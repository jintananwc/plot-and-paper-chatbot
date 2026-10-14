"""คุยกับผู้ให้บริการ LLM ด้วยฟอร์แมต OpenAI-compatible Chat Completions

ใช้ได้ทั้ง OpenRouter (ออนไลน์) และ LM Studio (รันในเครื่อง) — ต่างกันแค่ค่าใน .env
API แบบนี้เป็น stateless: ต้องส่ง history ทั้งหมดไปเองทุกครั้ง
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from config import Settings

log = logging.getLogger("chatbot")


class LLMError(Exception):
    """เรียกโมเดลไม่สำเร็จ — message เป็นภาษาไทยที่ปลอดภัยพอจะแสดงให้ผู้ใช้เห็น (ไม่มี stack trace)"""

    def __init__(self, message: str, code: str = "llm_error"):
        super().__init__(message)
        self.message = message
        self.code = code


_STATUS_MESSAGES = {
    400: ("llm_bad_request", "คำขอไปยังโมเดลไม่ถูกต้อง (อาจเป็นเพราะโมเดลนี้ไม่รองรับรูปภาพ)"),
    401: ("llm_auth", "API key ไม่ถูกต้องหรือหมดอายุ กรุณาตรวจสอบค่า LLM_API_KEY ใน .env"),
    402: ("llm_credit", "เครดิตของผู้ให้บริการ AI ไม่พอ กรุณาเติมเครดิตแล้วลองใหม่"),
    404: ("llm_model", "ไม่พบโมเดลที่ระบุ กรุณาตรวจสอบค่า LLM_MODEL ใน .env"),
    429: ("llm_rate_limit", "ถูกจำกัดอัตราการเรียกใช้งาน (rate limit) กรุณารอสักครู่แล้วลองใหม่"),
    503: ("llm_unavailable", "ผู้ให้บริการ AI ไม่พร้อมให้บริการตอนนี้ กรุณาลองใหม่ภายหลัง"),
}


def _timeout(settings: Settings) -> httpx.Timeout:
    # connect: ผูก TCP, write: ส่ง request, read: รอโทเค็นตอบกลับ, pool: รอ connection ว่าง
    return httpx.Timeout(connect=5.0, write=10.0, read=settings.request_timeout_seconds, pool=5.0)


def _headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.llm_api_key}"}


def _url(settings: Settings, path: str) -> str:
    return f"{settings.llm_base_url.rstrip('/')}/{path.lstrip('/')}"


def _require_config(settings: Settings) -> None:
    if not settings.llm_configured:
        raise LLMError(
            "ยังไม่ได้ตั้งค่า LLM_BASE_URL / LLM_API_KEY / LLM_MODEL ใน .env (ดูตัวอย่างใน .env.example)",
            code="llm_not_configured",
        )


def _status_error(status: int) -> LLMError:
    code, msg = _STATUS_MESSAGES.get(status, ("llm_upstream", f"ผู้ให้บริการ AI ตอบกลับด้วยข้อผิดพลาด ({status})"))
    return LLMError(msg, code=code)


def _connection_error(exc: Exception) -> LLMError:
    if isinstance(exc, httpx.TimeoutException):
        return LLMError("ผู้ให้บริการ AI ตอบสนองช้าเกินไป (timeout) กรุณาลองใหม่", code="llm_timeout")
    if isinstance(exc, httpx.ConnectError):
        return LLMError("เชื่อมต่อผู้ให้บริการ AI ไม่ได้ กรุณาตรวจสอบค่า LLM_BASE_URL "
                        "(ถ้าใช้ LM Studio ต้องเปิด local server ไว้)", code="llm_connection")
    return LLMError(f"เกิดข้อผิดพลาดในการเชื่อมต่อผู้ให้บริการ AI: {exc}", code="llm_connection")


def _payload(settings: Settings, messages: list[dict], model, temperature, max_tokens, stream=False) -> dict:
    payload: dict[str, Any] = {
        "model": model or settings.llm_model,
        "messages": messages,
        "temperature": settings.temperature if temperature is None else temperature,
    }

    # ปิด Thinking สำหรับโมเดลที่เรียกผ่าน OpenRouter
    if settings.llm_base_url.rstrip("/").startswith(
        "https://openrouter.ai/api/v1"
    ):
        payload["reasoning"] = {"enabled": False}

    if max_tokens or settings.max_tokens:
        payload["max_tokens"] = max_tokens or settings.max_tokens
    if stream:
        payload["stream"] = True
    return payload


async def chat_completion(
    client: httpx.AsyncClient,
    settings: Settings,
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """ส่งข้อความหนึ่งรอบแบบไม่สตรีม คืนข้อความตอบกลับ"""
    _require_config(settings)
    try:
        resp = await client.post(
            _url(settings, "chat/completions"), headers=_headers(settings),
            json=_payload(settings, messages, model, temperature, max_tokens), timeout=_timeout(settings),
        )
    except httpx.HTTPError as exc:
        raise _connection_error(exc) from exc

    if resp.status_code != 200:
        raise _status_error(resp.status_code)
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError("รูปแบบผลลัพธ์จากโมเดลไม่ถูกต้อง", code="llm_bad_response") from exc
    if not content or not str(content).strip():
        raise LLMError("โมเดลตอบกลับว่างเปล่า กรุณาลองใหม่", code="llm_empty")
    return str(content).strip()


async def chat_completion_stream(
    client: httpx.AsyncClient,
    settings: Settings,
    messages: list[dict],
    *,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """สตรีมคำตอบทีละชิ้น (SSE ของ Chat Completions) — yield ข้อความย่อย
    ถ้าเกิดข้อผิดพลาดจะโยน LLMError (ทั้งก่อนเริ่มและระหว่างสตรีม)"""
    _require_config(settings)
    try:
        async with client.stream(
            "POST", _url(settings, "chat/completions"), headers=_headers(settings),
            json=_payload(settings, messages, model, None, None, stream=True), timeout=_timeout(settings),
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                raise _status_error(resp.status_code)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                if raw == "[DONE]":  # สตรีมแบบ OpenAI-compatible จบด้วยบรรทัดนี้
                    return
                try:
                    event = json.loads(raw)
                except ValueError:
                    log.warning("ข้าม SSE data ที่ parse ไม่ได้: %r", raw[:100])
                    continue
                if event.get("error"):
                    raise LLMError(f"ผู้ให้บริการ AI แจ้งข้อผิดพลาดระหว่างตอบ: {event['error']}", code="llm_upstream")
                choices = event.get("choices") or []
                content = (choices[0].get("delta") or {}).get("content") if choices else None
                if content:
                    yield content
    except httpx.HTTPError as exc:
        raise _connection_error(exc) from exc


async def list_models(client: httpx.AsyncClient, settings: Settings) -> list[dict]:
    """รายชื่อโมเดลจากผู้ให้บริการ (เอนด์พอยต์ /models แบบ OpenAI-compatible คืน {"data": [...]})"""
    _require_config(settings)
    try:
        resp = await client.get(_url(settings, "models"), headers=_headers(settings), timeout=_timeout(settings))
    except httpx.HTTPError as exc:
        raise _connection_error(exc) from exc
    if resp.status_code != 200:
        raise _status_error(resp.status_code)
    try:
        return resp.json().get("data", [])
    except ValueError as exc:
        raise LLMError("รูปแบบผลลัพธ์รายชื่อโมเดลไม่ถูกต้อง", code="llm_bad_response") from exc
