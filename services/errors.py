"""รูปแบบ error เดียวกันทุก endpoint: {"error": {"code", "message"}, "request_id"}"""

import uuid

from fastapi.responses import JSONResponse


class ApiError(Exception):
    """โยนจาก router/service เมื่อข้อมูลเข้าไม่ถูกต้อง — main.py แปลงเป็น JSONResponse ให้"""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def error_response(status: int, code: str, message: str, request_id: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}, "request_id": request_id or new_request_id()},
    )
