from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# root ของโปรเจกต์ — ใช้หาไฟล์ prompts/knowledge/blocklist โดยไม่พึ่ง working directory
# (บน serverless เช่น Vercel ไม่การันตีว่าโปรเซสเริ่มจาก root ของโปรเจกต์)
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """ค่าตั้งทั้งหมดอ่านจาก .env (ดูตัวอย่างใน .env.example)

    ตัวแปรฝั่ง LLM ตั้งชื่อกลาง ๆ (LLM_*) เพราะโค้ดคุยด้วยฟอร์แมต OpenAI-compatible
    Chat Completions — สลับระหว่าง OpenRouter (ออนไลน์) กับ LM Studio (ในเครื่อง)
    ได้โดยแก้ .env อย่างเดียว ยังรับชื่อเดิม OPENROUTER_* / CHAT_MODEL ได้เช่นกัน
    """

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (OpenAI-compatible) ---
    llm_base_url: str = Field("https://openrouter.ai/api/v1",
                              validation_alias=AliasChoices("LLM_BASE_URL", "OPENROUTER_BASE_URL"))
    llm_api_key: str = Field("", validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY"))
    llm_model: str = Field("google/gemini-2.5-flash", validation_alias=AliasChoices("LLM_MODEL", "CHAT_MODEL"))
    vision_model: str = ""  # ว่าง = ใช้ LLM_MODEL (ต้องเป็นโมเดลที่รับภาพได้)
    temperature: float = 0.2
    max_tokens: int = 800
    request_timeout_seconds: float = 120.0  # เวลารอโทเค็นตอบกลับ (โมเดลในเครื่องอาจช้า)

    # --- Safety ---
    guard_enabled: bool = True
    guard_model: str = "meta-llama/llama-guard-4-12b"
    blocklist_path: str = "blocklist.txt"              # คำหยาบ/คำไม่สุภาพ (ตรวจทั้งข้อความผู้ใช้และคำตอบ)
    forbidden_words_path: str = "forbidden_words.txt"  # หัวข้อต้องห้ามของร้าน (ตรวจข้อความผู้ใช้)
    max_message_chars: int = 1000
    max_image_mb: float = 5.0
    max_history_turns: int = 6
    rate_limit_per_minute: int = 20

    # --- RAG ---
    knowledge_dir: str = "knowledge"           # ข้อมูลร้าน + นโยบาย (.md)
    books_csv: str = "knowledge/books.csv"     # ฐานข้อมูลหนังสือ
    rag_top_k: int = 4                         # จำนวนหัวข้อนโยบายที่แสดงเป็นแหล่งอ้างอิง
    rag_min_coverage: float = 0.3              # เกณฑ์ความเกี่ยวข้องขั้นต่ำ (0-1)
    # all = ส่งนโยบายทั้งหมดให้โมเดลทุกครั้ง (คลังเล็ก กันตกหล่น) | search = ส่งเฉพาะหัวข้อที่ค้นเจอ
    rag_policy_mode: str = "all"
    max_books_in_context: int = 20

    # --- อื่น ๆ ---
    prompts_dir: str = "prompts"
    admin_token: str = ""  # ใช้กับ POST /api/v1/knowledge/reload (ว่าง = ปิด endpoint นี้)
    cors_allowed_origins: str = "*"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @staticmethod
    def resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model.strip() or self.llm_model

    @property
    def llm_configured(self) -> bool:
        # LM Studio ไม่ต้องใช้ key จริง แต่ต้องมีค่าอะไรสักอย่าง (เช่น lm-studio)
        return bool(self.llm_api_key.strip() and self.llm_model.strip() and self.llm_base_url.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ใช้ object เดียวกันทั้งโปรเจกต์: from config import settings
settings = get_settings()
