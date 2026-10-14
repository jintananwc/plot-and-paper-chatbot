"""จุดเข้า (entrypoint) ที่ Vercel เรียกใช้

Vercel's Python runtime รันไฟล์ในโฟลเดอร์ api/ เป็น serverless function
ไฟล์นี้แค่ import FastAPI app ตัวจริงจาก main.py (ที่ root ของโปรเจกต์) มาเปิดใช้งาน
โค้ดแอปจริงทั้งหมดยังอยู่ที่ main.py / routers / services เหมือนตอนรันในเครื่อง
"""

import sys
from pathlib import Path

# เพิ่ม root ของโปรเจกต์เข้า sys.path ไม่งั้น "from main import app" จะหา main.py ไม่เจอ
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402,F401
