"""endpoint ฝั่งแชท: /api/v1/models, /chat, /chat/stream, /chat/reset"""

import json
import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from config import settings
from services.chat_pipeline import (
    EXPECTED_ERRORS, build_messages, error_from_exc, read_chat_request,
    step_guard_input, step_guard_output, step_retrieve, step_vision,
)
from services.errors import error_response
from services.llm_client import LLMError, chat_completion, chat_completion_stream, list_models
from services.sessions import SESSION_COOKIE, set_session_cookie
from services.state import sessions

log = logging.getLogger("chatbot")
router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/models")
async def get_models(request: Request):
    """รายชื่อโมเดลจากผู้ให้บริการ — proxy ฝั่งเซิร์ฟเวอร์ ไม่ส่ง API key ให้ frontend"""
    try:
        models = await list_models(request.app.state.http, settings)
    except LLMError as exc:
        return error_response(500 if exc.code == "llm_not_configured" else 502, exc.code, exc.message)
    return {"current": settings.llm_model, "count": len(models),
            "models": [{"id": m.get("id"), "name": m.get("name", m.get("id"))} for m in models]}


@router.post("/chat")
async def chat(request: Request, response: Response):
    """แชท (ข้อความ + รูป) ตอบครั้งเดียวเป็น JSON"""
    ctx = await read_chat_request(request)
    if ctx.new_session:
        set_session_cookie(response, ctx.session_id)
    try:
        if blocked := await step_guard_input(ctx):
            return blocked
        if ctx.image_bytes is not None and (blocked := await step_vision(ctx)):
            return blocked
        step_retrieve(ctx)
        reply = await chat_completion(ctx.client, settings, build_messages(ctx))
        ctx.timer.lap("llm")
        blocked, reply = await step_guard_output(ctx, reply)
        if blocked:
            return blocked
    except EXPECTED_ERRORS as exc:
        status, code, msg = error_from_exc(exc)
        log.error("[%s] %s: %s", ctx.request_id, code, msg)
        return error_response(status, code, msg, ctx.request_id)

    ctx.remember(reply)
    return ctx.answer(reply)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


STAGE_LABELS = {
    "guard_input": "กำลังตรวจความปลอดภัยของข้อความ",
    "vision": "กำลังวิเคราะห์รูปภาพ",
    "retrieval": "กำลังค้นคลังความรู้",
    "llm": "กำลังสร้างคำตอบ",
    "guard_output": "กำลังตรวจคำตอบ",
}


@router.post("/chat/stream")
async def chat_stream(request: Request):
    """สตรีมแบบ Server-Sent Events แต่ละบรรทัดเป็น `data: {json}` มี type ดังนี้
    stage (เริ่มขั้นตอน) | image_analysis | sources (เอกสารที่ใช้) | delta (ข้อความย่อย) |
    replace (แทนข้อความทั้งหมด เช่น คำตอบถูกบล็อก/ปิดบังข้อมูล) |
    done (result = ผลสรุปรูปแบบเดียวกับ /api/v1/chat) | error
    """
    ctx = await read_chat_request(request)  # error ส่วนตรวจข้อมูลเข้าตอบเป็น JSON ปกติ

    def stage(name: str) -> str:
        return _sse({"type": "stage", "stage": name, "label": STAGE_LABELS[name]})

    async def events():
        text = ""
        try:
            yield stage("guard_input")
            if blocked := await step_guard_input(ctx):
                yield _sse({"type": "done", "result": blocked})
                return
            if ctx.image_bytes is not None:
                yield stage("vision")
                if blocked := await step_vision(ctx):
                    yield _sse({"type": "done", "result": blocked})
                    return
                yield _sse({"type": "image_analysis", "image_analysis": ctx.image_analysis})
            yield stage("retrieval")
            step_retrieve(ctx)
            yield _sse({"type": "sources", "sources": ctx.sources()})

            yield stage("llm")
            async for chunk in chat_completion_stream(ctx.client, settings, build_messages(ctx)):
                text += chunk
                yield _sse({"type": "delta", "content": chunk})
            ctx.timer.lap("llm")
            if not text.strip():
                raise LLMError("โมเดลตอบกลับว่างเปล่า กรุณาลองใหม่", code="llm_empty")

            yield stage("guard_output")
            blocked, clean = await step_guard_output(ctx, text.strip())
            if blocked:
                # ข้อความถูกแสดงไปแล้วระหว่างสตรีม -> สั่งให้หน้าเว็บแทนที่ด้วยข้อความปฏิเสธ
                yield _sse({"type": "replace", "reply": blocked["reply"], "blocked": True})
                yield _sse({"type": "done", "result": blocked})
                return
            if clean != text.strip():
                yield _sse({"type": "replace", "reply": clean, "blocked": False})
            # บันทึกความจำเฉพาะเมื่อสตรีมจบสมบูรณ์ (ถ้าผู้ใช้กดหยุดกลางทาง จะไม่ถึงบรรทัดนี้)
            ctx.remember(clean)
            yield _sse({"type": "done", "result": ctx.answer(clean)})
        except EXPECTED_ERRORS as exc:
            status, code, msg = error_from_exc(exc)
            log.error("[%s] stream %s: %s", ctx.request_id, code, msg)
            yield _sse({"type": "error", "status": status, "code": code, "message": msg,
                        "request_id": ctx.request_id, "partial": bool(text)})
        except Exception:
            log.exception("[%s] stream failed", ctx.request_id)
            yield _sse({"type": "error", "status": 500, "code": "internal_error",
                        "message": "เกิดข้อผิดพลาดภายในระบบ กรุณาลองใหม่อีกครั้ง", "request_id": ctx.request_id})

    resp = StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    if ctx.new_session:
        set_session_cookie(resp, ctx.session_id)
    return resp


@router.post("/chat/reset")
async def chat_reset(request: Request, response: Response):
    """ล้างประวัติบทสนทนาของ session นี้ (ปุ่ม "เริ่มใหม่") โดยใช้ cookie เดิมต่อได้"""
    session_id, is_new = sessions.ensure(request.cookies.get(SESSION_COOKIE))
    if is_new:
        set_session_cookie(response, session_id)
    sessions.reset(session_id)
    return {"ok": True}
