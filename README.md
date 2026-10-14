# Plot & Paper Chatbot API — แชทบอทร้านเช่าหนังสือ

แชทบอทตอบคำถามลูกค้าของร้านเช่าหนังสือ **Plot & Paper** สร้างด้วย FastAPI + LLM แบบ OpenAI-compatible
ตอบจากฐานข้อมูลหนังสือ (20 เรื่อง) และนโยบายของร้านเท่านั้น พร้อมชั้นตรวจความปลอดภัยหลายชั้น

| ข้อมูลร้าน | |
|---|---|
| ประเภท | ร้านเช่าหนังสือ |
| เวลาทำการ | 10:00–20:00 |
| ติดต่อ | LINE @plotandpaper |
| ชำระเงิน | Cash / QR / Credit Card |

## ครอบคลุม Checklist ส่งมอบ 1

| หัวข้อ | สิ่งที่ทำในโปรเจกต์ | ไฟล์ |
|---|---|---|
| **LLM** | คุยกับโมเดลแบบ OpenAI-compatible (OpenRouter หรือ LM Studio), ส่งบริบท RAG + ประวัติแชท, สตรีมคำตอบ, แปลง error เป็นข้อความไทย | `services/llm_client.py`, `routers/chat.py` |
| **API** | 11 endpoint ตามแผนภาพสถาปัตยกรรม, error รูปแบบเดียวกันทุกจุด, รับทั้ง multipart และ JSON, Swagger ที่ `/docs` | `main.py`, `routers/`, `docs/architecture.md` |
| **UI** | แถบสถานะตามขั้นตอนจริงจากเซิร์ฟเวอร์ + ตัวจับเวลา, ข้อความทยอยขึ้นแบบสตรีม, ปุ่มหยุด, timeout, กล่อง error + ปุ่มลองใหม่, สถานะเซิร์ฟเวอร์, รายการหนังสือพร้อมสถานะ (คลิกเพื่อถาม), แหล่งอ้างอิง/เวลา | `static/index.html` |
| **RAG** | (1) ฐานข้อมูลหนังสือ: แยกเงื่อนไขจากคำถาม (ชื่อ/หมวด/สถานะ/ค่าเช่า/ภาษา/แพงที่สุด) แล้วกรองข้อมูลจริง + ค้นเชิงความหมายด้วย BM25 ภาษาไทย (2) นโยบายร้าน: ค้นหัวข้อที่เกี่ยวข้องเป็นแหล่งอ้างอิง (3) คำนวณค่าปรับ/ค่าเช่ารวมด้วยโค้ด ไม่ให้โมเดลคิดเลขเอง | `services/catalog.py`, `services/rag.py`, `knowledge/` |
| **Prompt** | กฎ MUST DO / MUST NOT ตามที่ร้านกำหนด แยกเป็นไฟล์, สคริปต์สุ่มทดสอบกฎ | `prompts/system.txt`, `eval/prompt_rules.json` |
| **Safety** | rate limit, จำกัดความยาว/ไฟล์, คำหยาบ, **หัวข้อต้องห้าม (ฆาตกรรม/murder) จับได้แม้แทรกช่องว่าง/เครื่องหมาย/แยกตัวอักษร/เลี่ยงด้วยความหมาย**, prompt injection, สั่งเปลี่ยนราคา/สร้างคูปอง/อนุมัติสิทธิ์/สมมติกฎใหม่, ขอข้อมูลลูกค้าคนอื่น, ปิดบัง PII, Llama Guard 4 ขาเข้า-ขาออก (fail-closed), canary กัน prompt รั่ว | `services/safety.py`, `forbidden_words.txt`, `blocklist.txt` |

## โครงสร้างไฟล์

มี `main.py` **ไฟล์เดียว** เป็นจุดเริ่มของแอป ส่วนอื่นแยกเป็น `routers/` (endpoint) และ `services/` (ตรรกะ)

```
plot-and-paper-chatbot/
├── main.py                  # จุดเริ่ม: สร้าง FastAPI app, CORS, หน้าเว็บ, /health, ตัวจัดการ error
├── config.py                # อ่านค่าตั้งจาก .env
├── routers/
│   ├── chat.py              # /api/v1/models, /chat, /chat/stream, /chat/reset
│   └── tools.py             # /api/v1/books, /guard/check, /rag/search, /knowledge, /knowledge/reload
├── services/
│   ├── chat_pipeline.py     # ไปป์ไลน์แชท: ตรวจข้อมูลเข้า → Safety → ภาพ → RAG → LLM → Safety
│   ├── catalog.py           # ค้นฐานข้อมูลหนังสือ + คำนวณค่าปรับ/ค่าเช่ารวม
│   ├── rag.py               # ค้นข้อความภาษาไทย (n-gram BM25) ใช้กับนโยบายและหนังสือ
│   ├── safety.py            # กฎความปลอดภัยทั้งหมด + Llama Guard
│   ├── categories.py        # ข้อความปฏิเสธภาษาไทยแต่ละกรณี + หมวด Llama Guard
│   ├── llm_client.py        # คุยกับ LLM: ปกติ / สตรีม / รายชื่อโมเดล
│   ├── prompts.py, sessions.py, state.py, errors.py
├── knowledge/
│   ├── 01_store_info.md     # ข้อมูลร้าน เวลา ติดต่อ ชำระเงิน
│   ├── 02_policies.md       # นโยบาย 5 ข้อ + ตารางค่าเช่าตามราคาปก
│   └── books.csv            # ฐานข้อมูลหนังสือ 20 เรื่อง
├── prompts/                 # system.txt (MUST DO / MUST NOT), vision_system.txt
├── forbidden_words.txt      # หัวข้อต้องห้าม (แก้ได้ทันที ไม่ต้องรีสตาร์ต)
├── blocklist.txt            # คำหยาบ
├── static/index.html        # หน้าแชท
├── api/index.py             # entrypoint สำหรับ Vercel
├── tests/                   # pytest 138 กรณี (จำลอง LLM ไม่ต้องใช้ key)
├── eval/                    # ชุดทดสอบ 30 ข้อ + กฎ 12 ข้อ + ภาพ 5 ภาพ สำหรับเอกสารส่งมอบ 2
├── docs/architecture.md     # แผนภาพสถาปัตยกรรม (Mermaid) + ตาราง endpoint
└── Dockerfile, vercel.json, requirements*.txt, pyproject.toml, .env.example
```

## ติดตั้งและรัน

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env             # แล้วใส่ค่า LLM_API_KEY ของตัวเอง
uvicorn main:app --reload        # หรือ: python main.py
```

เปิด http://127.0.0.1:8000 (หน้าแชท) และ http://127.0.0.1:8000/docs (Swagger)

### ตั้งค่า `.env` (ตัวที่สำคัญ)

| ตัวแปร | ตัวอย่าง | หมายเหตุ |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | ต้องจบด้วย `/v1` |
| `LLM_API_KEY` | (จาก dashboard ผู้ให้บริการ) | อย่า commit ค่านี้ขึ้น git |
| `LLM_MODEL` | `google/gemini-2.5-flash` | ควรรับภาพได้ ดูรายชื่อได้จาก `/api/v1/models` |
| `GUARD_ENABLED` / `GUARD_MODEL` | `true` / `meta-llama/llama-guard-4-12b` | ถ้าใช้ LM Studio ล้วน ให้ตั้ง `GUARD_ENABLED=false` |
| `BOOKS_CSV` | `knowledge/books.csv` | ฐานข้อมูลหนังสือ |
| `RAG_POLICY_MODE` | `all` | `all` = ส่งนโยบายทั้งหมดให้โมเดล (คลังเล็ก กันตกหล่น) |

ตัวแปรทั้งหมดดูใน `.env.example` · ชื่อโมเดลบน OpenRouter เปลี่ยนได้ ตรวจที่ https://openrouter.ai/models

## แก้ไขข้อมูลร้าน (ไม่ต้องแก้โค้ด)

| อยากแก้ | แก้ไฟล์ | มีผลเมื่อ |
|---|---|---|
| เพิ่ม/แก้หนังสือ, เปลี่ยนสถานะ | `knowledge/books.csv` (คอลัมน์เดิม) | รีสตาร์ต หรือ `POST /api/v1/knowledge/reload` |
| นโยบาย/ข้อมูลร้าน | `knowledge/*.md` (แบ่งหัวข้อด้วย `## `) | รีสตาร์ต หรือ reload |
| กฎ MUST DO / MUST NOT | `prompts/system.txt` | ทันที |
| หัวข้อต้องห้าม / คำหยาบ | `forbidden_words.txt` / `blocklist.txt` | ทันที |
| ข้อความปฏิเสธ | `services/categories.py` | รีสตาร์ต |

ตอนโหลด `books.csv` ระบบตรวจว่าค่าเช่าตรงกับตารางราคาปกหรือไม่ ถ้าไม่ตรงจะแสดงใน `/health` → `books.data_warnings`

## API โดยย่อ

| Method | Path | หน้าที่ |
|---|---|---|
| GET | `/health` | สถานะระบบ/การตั้งค่า + จำนวนหนังสือ |
| GET | `/api/v1/models` | รายชื่อโมเดลจากผู้ให้บริการ (ไม่ส่ง key ออกไป) |
| POST | `/api/v1/chat` | แชท ตอบครั้งเดียวเป็น JSON |
| POST | `/api/v1/chat/stream` | แชทแบบสตรีม (SSE) พร้อมสถานะแต่ละขั้นตอน |
| POST | `/api/v1/chat/reset` | ล้างความจำบทสนทนาของ session |
| GET | `/api/v1/books` | รายการหนังสือ (`?status=` `?category=` `?max_rental=` `?q=`) |
| POST | `/api/v1/guard/check` | ตรวจความปลอดภัยอย่างเดียว |
| POST | `/api/v1/rag/search` | ค้นนโยบายร้านอย่างเดียว |
| GET | `/api/v1/knowledge` | รายการเอกสารในคลังความรู้ |
| POST | `/api/v1/knowledge/reload` | โหลดคลังความรู้และหนังสือใหม่ (ต้องมี `X-Admin-Token`) |

### `POST /api/v1/chat`
- **multipart/form-data:** `message`, `history` (JSON string, ไม่บังคับ), `image` (.jpg/.png/.webp, ไม่บังคับ)
- **application/json:** `{"message": "...", "history": [...], "image_base64": "...", "image_mime": "image/jpeg"}`

ผลตอบกลับ (200)
```json
{
  "request_id": "3a55f1714012",
  "type": "answer",
  "reply": "Atomic Habits มีให้เช่าค่ะ สถานะ Available ค่าเช่า 7 วัน 50 บาท",
  "sources": [{"id": "B001", "doc": "books.csv", "title": "Atomic Habits", "heading": "Self-Development · Available · ค่าเช่า 50 บาท"}],
  "image_analysis": null,
  "safety": {"input": {"result": "safe", "layer": "pass", "categories": [], "redacted": []}, "output": {"result": "safe"}},
  "timings_ms": {"guard_input": 410.2, "retrieval": 1.3, "llm": 1820.5, "guard_output": 390.1, "total": 2625.0}
}
```
`type = "blocked"` เมื่อชั้น Safety ปฏิเสธ — `safety.input.layer` บอกชั้นที่ทำงาน:
`blocklist` (คำหยาบ) · `forbidden_word` (หัวข้อต้องห้าม) · `injection` · `policy_override` · `privacy` · `llama_guard` · `vision`

`/api/v1/chat/stream` ส่งทีละบรรทัด `data: {json}` — type: `stage`, `image_analysis`, `sources`, `delta`, `replace`, `done` (`result` รูปแบบเดียวกับข้างบน), `error`

Error ทุก endpoint: `{"error": {"code": "...", "message": "ข้อความไทย"}, "request_id": "..."}`
(400 ข้อมูลไม่ครบ · 413 ยาว/ใหญ่เกิน · 415 ไฟล์ไม่ใช่ภาพ · 422 รูปแบบผิด · 429 ส่งถี่เกิน · 500 ยังไม่ตั้งค่า LLM · 502 เรียกโมเดลไม่สำเร็จ · 503 ตรวจความปลอดภัยไม่ได้)

### ตัวอย่าง curl
```bash
curl -X POST http://localhost:8000/api/v1/chat -F "message=Atomic Habits มีให้เช่าไหม และค่าเช่าเท่าไหร่?"
curl -N -X POST http://localhost:8000/api/v1/chat/stream -F "message=ฉันเช่า 2 เล่มและคืนช้า 4 วัน ต้องเสียค่าปรับรวมเท่าไหร่?"
curl -X POST http://localhost:8000/api/v1/chat -F "message=หนังสือเล่มนี้มีให้เช่าไหม" -F "image=@eval/images/1_book_tag_atomic_habits.jpg"
curl "http://localhost:8000/api/v1/books?status=Available&max_rental=50"
curl -X POST http://localhost:8000/api/v1/guard/check -H "Content-Type: application/json" -d '{"message":"มีนิยายแนว m.u.r.d.e.r ไหม","direction":"input"}'
```

## การทดสอบ

### 1) Unit/API test (ไม่ต้องใช้ key)
```bash
pytest -q        # 138 passed
```
ครอบคลุม: ทุก endpoint, ฐานข้อมูลหนังสือกับคำถามหมวด A–C, การคำนวณค่าปรับ, การค้นนโยบาย, **คำถามหมวด D–E ทั้ง 12 ข้อต้องถูกบล็อก และคำถามปกติ 27 ข้อต้องไม่ถูกบล็อกผิด**, สตรีม, session, error ทุกแบบ

### 2) ชุดทดสอบกับโมเดลจริง → ตารางสำหรับเอกสารส่งมอบ 2
```bash
python eval/make_sample_images.py     # สร้างภาพทดสอบ 1, 2, 4, 5
# ถ่ายภาพหนังสือเสียหายเอง: eval/images/3_damaged_book.jpg
uvicorn main:app                      # เปิดเซิร์ฟเวอร์ไว้อีกหน้าต่าง
python eval/run_eval.py               # หรือ --suite cases|prompt|images, --group A
```
ได้ไฟล์ `eval/results/report_<เวลา>.md` (ตาราง: ผลตอบ + แหล่งข้อมูล + ผ่าน/ไม่ผ่าน + เวลาตอบ) และ `.csv`

| ชุด | ไฟล์ | จำนวน |
|---|---|---|
| คำถามทดสอบ หมวด A–E | `eval/test_cases.json` | 30 ข้อ |
| กฎ MUST DO / MUST NOT | `eval/prompt_rules.json` | 12 ข้อ (สุ่มได้ด้วย `--sample`) |
| ภาพทดสอบ | `eval/images.json` + `eval/images/` | 5 ภาพ |

> เกณฑ์ผ่าน/ไม่ผ่านอัตโนมัติใช้คำสำคัญ ควรให้คนในทีมอ่านตรวจซ้ำก่อนส่ง

## Deploy

**Vercel:** push ขึ้น GitHub → New Project → ใส่ Environment Variables ตาม `.env` → Deploy (ใช้ `vercel.json` + `api/index.py`)
ข้อจำกัด: ความจำ session และ rate limit อยู่ในหน่วยความจำของแต่ละ instance (หน้า UI ส่ง history เองจึงยังจำบทสนทนาได้)

**Docker / Render:**
```bash
docker build -t plot-and-paper-chatbot .
docker run -p 8000:8000 --env-file .env plot-and-paper-chatbot
```

## ข้อควรรู้
- **หัวข้อต้องห้ามตรวจเฉพาะข้อความลูกค้า:** หนังสือ *The Midnight Library Murder* (B016) ยังอยู่ในฐานข้อมูล บอทจึงยังแนะนำได้ถ้าลูกค้าถามแบบกว้าง ๆ เช่น "นิยายสืบสวนมีอะไรบ้าง" ถ้าไม่ต้องการให้แนะนำเล่มนี้เลย ให้ลบออกจาก `books.csv` หรือเปลี่ยนสถานะ
- คำนวณค่าปรับด้วยโค้ดตามสูตร `จำนวนเล่ม × วันที่ช้า × 10 บาท` แล้วส่งผลให้โมเดล จึงไม่คิดเลขผิด
- ร้านไม่มีข้อมูลค่าชดเชยหนังสือหาย และไม่มีค่าต่ออายุ บอทจึงตอบให้ติดต่อร้าน (ไม่แต่งตัวเลข)
- แบบสตรีม คำตอบจะแสดงก่อนตรวจขาออกเสร็จ ถ้าไม่ผ่าน หน้าเว็บจะแทนด้วยข้อความปฏิเสธทันที
