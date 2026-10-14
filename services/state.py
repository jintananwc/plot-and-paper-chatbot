"""object ที่ใช้ร่วมกันทั้งแอป (สร้างครั้งเดียวตอนเริ่มโปรแกรม)"""

from config import settings
from services.catalog import Catalog
from services.rag import KnowledgeBase
from services.safety import RateLimiter, SafetyLayer
from services.sessions import SessionStore

kb = KnowledgeBase(str(settings.resolve(settings.knowledge_dir)))      # ข้อมูลร้าน + นโยบาย
catalog = Catalog(str(settings.resolve(settings.books_csv)))          # ฐานข้อมูลหนังสือ
safety = SafetyLayer(settings)
limiter = RateLimiter(settings.rate_limit_per_minute)
sessions = SessionStore(max_messages=settings.max_history_turns * 2)
