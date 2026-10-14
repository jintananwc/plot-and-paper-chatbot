"""endpoint เสริม: ข้อมูลหนังสือ, ตรวจความปลอดภัยอย่างเดียว, ค้นคลังความรู้อย่างเดียว, จัดการคลังความรู้"""

import logging
import time
from typing import Literal

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, Field

from config import settings
from services.errors import ApiError
from services.safety import GuardUnavailable
from services.catalog import CATEGORY_ALIASES
from services.state import catalog, kb, safety

log = logging.getLogger("chatbot")
router = APIRouter(prefix="/api/v1", tags=["tools"])


class GuardRequest(BaseModel):
    message: str = Field(min_length=1)
    direction: Literal["input", "output"] = "input"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


@router.get("/books")
def list_books(
    q: str | None = Query(None, description="ค้นแบบข้อความ เช่น 'หนังสือเกี่ยวกับ AI' หรือ 'ค่าเช่าไม่เกิน 50 บาท'"),
    category: str | None = Query(None, description="เช่น Finance, Psychology"),
    status: str | None = Query(None, description="Available | Rented | Reserved"),
    max_rental: int | None = Query(None, ge=0, description="ค่าเช่า 7 วันไม่เกิน (บาท)"),
):
    """รายการหนังสือในร้าน กรองได้ตามหมวด สถานะ ค่าเช่า หรือค้นด้วยข้อความ"""
    if status and status not in ("Available", "Rented", "Reserved"):
        raise ApiError(422, "invalid_status", "status ต้องเป็น Available, Rented หรือ Reserved")
    if category and category not in CATEGORY_ALIASES:
        raise ApiError(422, "invalid_category", "category ต้องเป็นหนึ่งใน: " + ", ".join(CATEGORY_ALIASES))
    if q:
        res = catalog.search(q)
        books, facts = res.books, res.facts
    else:
        books, facts = catalog.books, []
    if category or status or max_rental is not None:
        price = {"field": "rental", "op": "<=", "value": max_rental} if max_rental is not None else None
        allowed = {b.book_id for b in catalog.filter(categories=[category] if category else None,
                                                      status=status, price=price)}
        books = [b for b in books if b.book_id in allowed]
    return {"count": len(books), "facts": facts, "books": [b.to_dict() for b in books],
            "categories": list(CATEGORY_ALIASES)}


@router.post("/guard/check")
async def guard_check(req: GuardRequest, request: Request):
    """ตรวจข้อความด้วยชั้น Safety (input = ข้อความผู้ใช้, output = คำตอบของบอท)"""
    t0 = time.perf_counter()
    client = request.app.state.http
    try:
        if req.direction == "input":
            result, _ = await safety.check_input(client, req.message)
        else:
            result, _ = await safety.check_output(client, req.message)
    except GuardUnavailable as exc:
        log.error("guard failed: %s", exc)
        raise ApiError(503, "guard_unavailable", "ระบบตรวจความปลอดภัยไม่พร้อมใช้งานชั่วคราว กรุณาลองใหม่") from exc
    return {
        **result.to_dict(),
        "msg": req.message if result.safe else result.message,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@router.post("/rag/search")
def rag_search(req: SearchRequest):
    """ค้นคลังความรู้อย่างเดียว — ดูว่าคำถามนี้ระบบจะดึงเอกสารไหนมาใช้ตอบ"""
    t0 = time.perf_counter()
    hits = kb.search(req.query, top_k=req.top_k, min_coverage=0)
    return {
        "query": req.query,
        "min_coverage": settings.rag_min_coverage,
        "results": [{**h.to_dict(), "used_for_answer": h.coverage >= settings.rag_min_coverage} for h in hits],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@router.get("/knowledge")
def list_knowledge():
    """รายการเอกสารในคลังความรู้ (ข้อมูลร้าน/นโยบาย) และจำนวนหนังสือในฐานข้อมูล"""
    return {"documents": kb.summary(), "total_chunks": len(kb.chunks),
            "books": {"file": settings.books_csv, "total": len(catalog.books), "data_warnings": catalog.warnings}}


@router.post("/knowledge/reload")
def reload_knowledge(x_admin_token: str | None = Header(default=None)):
    """โหลดคลังความรู้และฐานข้อมูลหนังสือใหม่หลังแก้ไฟล์ใน knowledge/ (ต้องส่ง header X-Admin-Token)"""
    if not settings.admin_token:
        raise ApiError(403, "reload_disabled", "ยังไม่ได้ตั้งค่า ADMIN_TOKEN จึงปิดการใช้งาน endpoint นี้")
    if x_admin_token != settings.admin_token:
        raise ApiError(401, "unauthorized", "X-Admin-Token ไม่ถูกต้อง")
    kb.load()
    catalog.load()
    return {"status": "reloaded", "documents": len(kb.summary()), "chunks": len(kb.chunks),
            "books": len(catalog.books), "data_warnings": catalog.warnings}
