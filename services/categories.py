"""หมวดของ Llama Guard 4 และข้อความปฏิเสธภาษาไทยของร้าน Plot & Paper

แก้ blocked / message ได้ที่นี่ที่เดียว
- blocked=False  -> ถ้าโมเดลตอบ unsafe เฉพาะหมวดนี้ จะถือว่า safe
- ข้อความปฏิเสธต้องมีเฉพาะหมวดที่ blocked=True
"""

from dataclasses import dataclass

_TAIL = " หากต้องการสอบถามเรื่องหนังสือหรือการเช่าของร้าน Plot & Paper ยินดีช่วยเหลือค่ะ"


@dataclass(frozen=True)
class Category:
    code: str
    name: str
    blocked: bool
    message: str | None = None


_CATEGORIES = [
    Category("S1", "Violent Crimes", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหาที่เกี่ยวข้องกับความรุนแรงหรืออาชญากรรมรุนแรงได้" + _TAIL),
    Category("S2", "Non-Violent Crimes", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหาที่เกี่ยวข้องกับการกระทำผิดกฎหมายได้" + _TAIL),
    Category("S3", "Sex-Related Crimes", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหาที่เกี่ยวข้องกับอาชญากรรมทางเพศได้" + _TAIL),
    Category("S4", "Child Sexual Exploitation", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหานี้ได้"),
    Category("S5", "Defamation", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหาที่อาจเป็นการใส่ร้ายหรือหมิ่นประมาทบุคคลหรือองค์กรได้"),
    # ร้านมีหนังสือหมวดการเงิน/จิตวิทยา คำถามเรื่องหนังสือเหล่านี้ไม่ใช่การขอคำแนะนำเฉพาะทาง
    # จึงไม่บล็อกหมวดนี้ (กันบล็อกผิด) — system prompt ยังห้ามให้คำแนะนำการเงิน/การแพทย์โดยตรงอยู่
    Category("S6", "Specialized Advice", False),
    Category("S7", "Privacy", True,
             "ขออภัยค่ะ ไม่สามารถให้หรือเก็บข้อมูลส่วนบุคคลของผู้อื่นได้ เพื่อความเป็นส่วนตัวของทุกคนค่ะ"),
    Category("S8", "Intellectual Property", False),
    Category("S9", "Indiscriminate Weapons", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหาที่เกี่ยวข้องกับอาวุธที่ก่อความเสียหายวงกว้างได้"),
    Category("S10", "Hate", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหาที่แสดงความเกลียดชังหรือเลือกปฏิบัติต่อบุคคลหรือกลุ่มใดได้"),
    Category("S11", "Suicide & Self-Harm", True,
             "ขออภัยค่ะ เรื่องนี้แชทบอทช่วยตอบไม่ได้ แต่ถ้าคุณกำลังรู้สึกไม่ไหว โปรดคุยกับคนใกล้ชิด "
             "หรือโทรสายด่วนสุขภาพจิต 1323 (ตลอด 24 ชั่วโมง) นะคะ"),
    Category("S12", "Sexual Content", True,
             "ขออภัยค่ะ ไม่สามารถตอบเนื้อหาทางเพศได้" + _TAIL),
    Category("S13", "Elections", False),
    Category("S14", "Code Interpreter Abuse", False),
]

CATEGORIES: dict[str, Category] = {c.code: c for c in _CATEGORIES}

# ---------------------------------------------------------------------------
# ชั้นตรวจแบบกฎตายตัว (ไม่บอกผู้ใช้ว่าติดคำไหน/กฎไหน เพื่อไม่ให้หาทางเลี่ยงได้ง่าย)
# ---------------------------------------------------------------------------

BLOCKLIST_CODE = "BLOCKLIST"
BLOCKLIST_MESSAGE = "ขออภัยค่ะ ไม่สามารถตอบข้อความนี้ได้ กรุณาใช้ถ้อยคำที่สุภาพ" + _TAIL

FORBIDDEN_CODE = "FORBIDDEN_TOPIC"
FORBIDDEN_MESSAGE = ("ขออภัยค่ะ ร้านไม่สามารถให้ข้อมูลหรือแนะนำหนังสือในหัวข้อนี้ได้ "
                     "หากสนใจหนังสือหมวดอื่น เช่น พัฒนาตนเอง การเงิน จิตวิทยา หรือเทคโนโลยี ยินดีช่วยค้นหาให้ค่ะ")

INJECTION_CODE = "PROMPT_INJECTION"
INJECTION_MESSAGE = ("ขออภัยค่ะ ไม่สามารถทำตามคำสั่งที่ให้ละเลยกฎ เปลี่ยนบทบาท หรือเปิดเผยคำสั่งภายในของระบบได้"
                     + _TAIL)

OVERRIDE_CODE = "POLICY_OVERRIDE"
OVERRIDE_MESSAGE = ("ขออภัยค่ะ แชทบอทไม่สามารถเปลี่ยนค่าเช่า สร้างส่วนลดหรือคูปอง อนุมัติสิทธิ์พิเศษ "
                    "หรือสมมติกฎใหม่ได้ ค่าเช่าและนโยบายเป็นไปตามที่ร้าน Plot & Paper กำหนดเท่านั้น "
                    "(เช่น ระยะเวลาเช่ามาตรฐาน 7 วัน) สอบถามเพิ่มเติมได้ทาง LINE @plotandpaper ค่ะ")

PRIVACY_CODE = "CUSTOMER_PRIVACY"
PRIVACY_MESSAGE = ("ขออภัยค่ะ ร้านไม่สามารถเปิดเผยข้อมูลของลูกค้าท่านอื่นได้ เพื่อความเป็นส่วนตัวของลูกค้าทุกท่าน "
                   "หากต้องการทราบสถานะหนังสือ (Available / Rented / Reserved) สอบถามได้เลยค่ะ")

PROMPT_LEAK_CODE = "PROMPT_LEAK"
PROMPT_LEAK_MESSAGE = "ขออภัยค่ะ ไม่สามารถให้ข้อมูลส่วนนี้ได้" + _TAIL

IMAGE_UNSAFE_CODE = "IMAGE_UNSAFE"
IMAGE_UNSAFE_MESSAGE = "ขออภัยค่ะ ภาพนี้มีเนื้อหาที่ไม่เหมาะสม ระบบจึงไม่สามารถประมวลผลได้ค่ะ"
