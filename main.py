"""จุดเริ่มต้นของแอป (ไฟล์ main มีไฟล์เดียว)

รันได้ 2 แบบ:
    uvicorn main:app --reload      # ตอนพัฒนา (รีโหลดเองเมื่อแก้โค้ด)
    python main.py                 # อ่าน HOST / PORT จาก .env

โครงสร้าง:
    main.py              สร้าง FastAPI app, CORS, หน้าเว็บ, /health, ตัวจัดการ error
    config.py            อ่านค่าตั้งจาก .env
    routers/chat.py      /api/v1/models, /chat, /chat/stream, /chat/reset
    routers/tools.py     /api/v1/books, /guard/check, /rag/search, /knowledge, /knowledge/reload
    services/            ตรรกะทั้งหมด (LLM, RAG, Safety, ไปป์ไลน์แชท, session)
    api/index.py         entrypoint สำหรับ Vercel (import app จากไฟล์นี้)
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, settings
from routers.chat import router as chat_router
from routers.tools import router as tools_router
from services.errors import ApiError, error_response
from services.state import catalog, kb

STATIC_DIR = BASE_DIR / "static"

# ตั้ง logging กลางไว้ที่จุดเริ่มโปรแกรม ให้ log จากทุกโมดูลออกมาในรูปแบบเดียวกัน
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ใช้ HTTP client ตัวเดียวทั้งแอป (เปิด connection ซ้ำได้ เร็วกว่าสร้างใหม่ทุกครั้ง)
    app.state.http = httpx.AsyncClient()
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="Plot & Paper Chatbot API",
    description="แชทบอทร้านเช่าหนังสือ Plot & Paper — ตอบจากฐานข้อมูลหนังสือและนโยบายร้าน (RAG ภาษาไทย) พร้อมชั้นตรวจความปลอดภัย",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials="*" not in settings.cors_origins_list,  # ใช้ cookie ข้ามโดเมนได้เฉพาะเมื่อระบุ origin ชัดเจน
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(chat_router)
app.include_router(tools_router)


# ---------------------------------------------------------------------------
# ตัวจัดการ error — ทุก endpoint ตอบรูปแบบเดียวกัน {"error": {"code", "message"}, "request_id"}
# ---------------------------------------------------------------------------

@app.exception_handler(ApiError)
async def handle_api_error(_: Request, exc: ApiError):
    return error_response(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    return error_response(422, "validation_error", f"ข้อมูลที่ส่งมาไม่ถูกต้อง ({where}: {first.get('msg', '')})")


@app.exception_handler(Exception)
async def handle_unexpected(_: Request, exc: Exception):
    log.exception("unexpected error")
    return error_response(500, "internal_error", "เกิดข้อผิดพลาดภายในระบบ กรุณาลองใหม่อีกครั้ง")


# ---------------------------------------------------------------------------
# หน้าเว็บและสถานะระบบ
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    # เช็คว่าเซิร์ฟเวอร์ทำงานและตั้งค่าครบ (ไม่ได้ยิงไปหาผู้ให้บริการ AI จริง)
    return {
        "status": "ok",
        "llm_configured": settings.llm_configured,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "vision_model": settings.effective_vision_model,
        "guard_enabled": settings.guard_enabled,
        "guard_model": settings.guard_model if settings.guard_enabled else None,
        "knowledge": {"documents": len(kb.summary()), "chunks": len(kb.chunks)},
        "books": {"total": len(catalog.books), "data_warnings": catalog.warnings},
    }


if __name__ == "__main__":
    import os

    import uvicorn

    # Render/Railway กำหนดพอร์ตผ่าน PORT — ใช้ค่านั้นก่อนถ้ามี
    uvicorn.run("main:app", host=settings.host, port=int(os.getenv("PORT", settings.port)), log_level=settings.log_level)
