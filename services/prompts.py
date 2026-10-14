"""โหลด prompt จากไฟล์ .txt — อ่านใหม่ทุกครั้ง แก้ไฟล์แล้วมีผลทันทีไม่ต้องรีสตาร์ต"""

from config import settings

# รหัสลับที่ฝังไว้ใน system prompt — ถ้าคำตอบมีรหัสนี้ แปลว่า prompt รั่ว (ใช้ใน output guard)
PROMPT_CANARY = "PLOTPAPER-SYS-5C2E"


def load_prompt(name: str) -> str:
    path = settings.resolve(settings.prompts_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ prompt: {path}")
    return path.read_text(encoding="utf-8").replace("{{CANARY}}", PROMPT_CANARY)
