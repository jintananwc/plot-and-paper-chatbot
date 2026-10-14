"""RAG ภาษาไทยแบบไม่ต้องพึ่ง embedding API

- อ่านไฟล์ .md ในโฟลเดอร์ knowledge/ แล้วตัดเป็นชิ้น (chunk) ตามหัวข้อ "## "
- ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ จึงตัดเป็น character n-gram (2-3 ตัวอักษร)
  แทนการตัดคำ ทำให้ไม่ต้องติดตั้งตัวตัดคำและทนต่อคำสะกดต่างกันเล็กน้อย
- จัดอันดับด้วย BM25 แล้วกรองด้วย "coverage" = สัดส่วนน้ำหนัก (IDF) ของคำค้นที่พบในชิ้นนั้น
  ถ้าต่ำกว่าเกณฑ์ถือว่าคลังความรู้ไม่มีคำตอบ -> บอทจะบอกว่าไม่มีข้อมูล แทนการเดา
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*|[฀-๿]+")
# เครื่องหมายที่ไม่ช่วยในการค้น (ไม้ยมก, ไปยาลน้อย ฯลฯ)
_THAI_NOISE = str.maketrans("", "", "ๆฯ")

# คำเติมที่พบบ่อยในคำถาม แต่ไม่บอกเนื้อหา — ตัดออกก่อนค้น
_THAI_STOPWORDS = [
    "ครับ", "ค่ะ", "คะ", "นะคะ", "นะครับ", "หน่อย", "ไหม", "มั้ย", "บ้าง", "อะไร",
    "อย่างไร", "ยังไง", "เท่าไหร่", "เท่าไร", "หรือเปล่า", "หรือไม่", "ได้ไหม",
    "ขอถาม", "อยากทราบ", "อยากรู้", "สอบถาม", "ช่วย", "บอก",
    "หนังสือ", "ร้าน", "แนะนำ", "ฉัน", "คุณ", "ให้", "มี", "ที่", "และ", "ต้อง",
]
_STOP_RE = re.compile("|".join(sorted(map(re.escape, _THAI_STOPWORDS), key=len, reverse=True)))


# คำพูดทั่วไปของลูกค้า -> คำที่ใช้ในคลังความรู้ (เพิ่มได้ตามธุรกิจ)
SYNONYMS: dict[str, str] = {
    "จ่ายเงิน": "ชำระเงิน", "จ่ายตัง": "ชำระเงิน", "โอนเงิน": "ชำระเงิน qr", "สแกน": "ชำระเงิน qr",
    "บัตรเครดิต": "ชำระเงิน credit card", "เงินสด": "ชำระเงิน cash",
    "กี่โมง": "เวลาทำการ เปิด", "เปิดกี่": "เวลาทำการ", "ปิดกี่": "เวลาทำการ", "ร้านเปิด": "เวลาทำการ",
    "ติดต่อ": "ช่องทางติดต่อ line", "ไลน์": "line",
    "นานกี่วัน": "ระยะเวลาเช่ามาตรฐาน", "กี่วัน": "ระยะเวลา วัน", "ยืม": "เช่า",
    "กี่เล่ม": "สูงสุด เล่มต่อครั้ง", "สูงสุด": "สูงสุด เล่มต่อครั้ง",
    "คืนช้า": "คืนหนังสือเกินกำหนด ค่าปรับ late return", "ส่งคืนช้า": "คืนหนังสือเกินกำหนด ค่าปรับ",
    "เลยกำหนด": "เกินกำหนด ค่าปรับ", "ค่าปรับ": "ค่าปรับ คืนหนังสือเกินกำหนด",
    "ต่ออายุ": "ต่ออายุการเช่า renewal", "ต่อเวลา": "ต่ออายุการเช่า", "ขยายเวลา": "ต่ออายุการเช่า",
    "หาย": "สูญหาย ค่าชดเชย", "ทำหาย": "สูญหาย ค่าชดเชย", "ขาด": "เสียหาย", "ชำรุด": "เสียหาย",
    "เปียก": "เสียหาย", "พัง": "เสียหาย", "ค่าชดเชย": "สูญหาย เสียหาย ค่าชดเชย",
    "จอง": "การจอง reservation สถานะ", "คิว": "จองคิว rented",
    "ราคาปก": "อัตราค่าเช่า ราคาปก", "คิดค่าเช่า": "อัตราค่าเช่า ราคาปก",
}
_SYN_RE = re.compile("|".join(sorted(map(re.escape, SYNONYMS), key=len, reverse=True)))


def expand_query(text: str) -> str:
    extra = {SYNONYMS[m] for m in _SYN_RE.findall(text)}
    return f"{text} {' '.join(extra)}" if extra else text


# สระบน/ล่างและวรรณยุกต์ — n-gram ที่ขึ้นต้นด้วยอักษรเหล่านี้ไม่มีความหมาย ตัดทิ้งลดสัญญาณรบกวน
_COMBINING = set("\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e")


def _is_thai(s: str) -> bool:
    return "฀" <= s[0] <= "๿"


def tokenize(text: str, *, query: bool = False, extra_stopwords: list[str] | None = None) -> list[str]:
    text = text.casefold().translate(_THAI_NOISE)
    if query:
        text = _STOP_RE.sub(" ", text)
        for w in sorted(extra_stopwords or [], key=len, reverse=True):
            text = text.replace(w, " ")
    tokens: list[str] = []
    for part in _TOKEN_RE.findall(text):
        if _is_thai(part):
            if len(part) < 2:
                continue
            for n in (2, 3):
                tokens.extend(g for i in range(len(part) - n + 1) if (g := part[i : i + n])[0] not in _COMBINING)
        else:
            tokens.append(part.replace(",", ""))
    return tokens


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    heading: str
    text: str
    tf: Counter = field(default_factory=Counter, repr=False)
    length: int = 0

    def as_context(self) -> str:
        return f"[{self.id}] {self.title} › {self.heading}\n{self.text}"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    coverage: float

    def to_dict(self) -> dict:
        return {
            "id": self.chunk.id,
            "doc": self.chunk.doc,
            "title": self.chunk.title,
            "heading": self.chunk.heading,
            "score": round(self.score, 3),
            "coverage": round(self.coverage, 3),
            "text": self.chunk.text,
        }


def split_markdown(doc_name: str, raw: str) -> list[Chunk]:
    """แบ่งเอกสารตาม '## ' โดยใช้ '# ' บรรทัดแรกเป็นชื่อเอกสาร"""
    lines = raw.splitlines()
    title = doc_name
    for ln in lines:
        if ln.startswith("# "):
            title = ln[2:].strip()
            break

    chunks: list[Chunk] = []
    heading, buf = "ภาพรวม", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            idx = len(chunks) + 1
            chunks.append(Chunk(f"{Path(doc_name).stem}#{idx}", doc_name, title, heading, body))

    for ln in lines:
        if ln.startswith("# "):
            continue
        if ln.startswith("## "):
            flush()
            heading, buf = ln[3:].strip(), []
        else:
            buf.append(ln)
    flush()
    return chunks


class KnowledgeBase:
    def __init__(self, directory: str, k1: float = 1.5, b: float = 0.75):
        self.directory = Path(directory)
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.idf: dict[str, float] = {}
        self.avg_len = 0.0
        self.max_idf = 1.0
        self.load()

    # ---------- สร้างดัชนี ----------
    def load(self) -> None:
        chunks: list[Chunk] = []
        if self.directory.exists():
            for path in sorted(self.directory.glob("*.md")):
                chunks.extend(split_markdown(path.name, path.read_text(encoding="utf-8-sig")))
        self.set_chunks(chunks)

    def set_chunks(self, chunks: list["Chunk"]) -> None:
        """สร้างดัชนี BM25 จากชิ้นข้อมูล (ใช้ร่วมกับฐานข้อมูลหนังสือได้)"""
        df: Counter = Counter()
        for c in chunks:
            # ให้หัวข้อมีน้ำหนักด้วย จึงรวมชื่อเรื่อง+หัวข้อเข้าไปตอนนับคำ
            toks = tokenize(f"{c.title} {c.heading} {c.heading} {c.text}")
            c.tf = Counter(toks)
            c.length = len(toks)
            df.update(c.tf.keys())

        n = len(chunks) or 1
        self.chunks = chunks
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.max_idf = max(self.idf.values(), default=1.0)
        self.avg_len = (sum(c.length for c in chunks) / len(chunks)) if chunks else 0.0

    # ---------- ค้นหา ----------
    def search(self, query: str, top_k: int = 4, min_coverage: float = 0.0,
               extra_stopwords: list[str] | None = None) -> list[SearchHit]:
        q_terms = set(tokenize(expand_query(query), query=True, extra_stopwords=extra_stopwords))
        if not q_terms or not self.chunks:
            return []
        # คำที่ไม่อยู่ในคลังเลย ได้น้ำหนักสูงสุด -> ดึง coverage ลง (คำถามนอกเรื่อง)
        weights = {t: self.idf.get(t, self.max_idf) for t in q_terms}
        total_w = sum(weights.values()) or 1.0

        hits: list[SearchHit] = []
        for c in self.chunks:
            score, matched_w = 0.0, 0.0
            for t in q_terms:
                f = c.tf.get(t, 0)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * c.length / (self.avg_len or 1))
                score += idf * f * (self.k1 + 1) / denom
                matched_w += weights[t]
            if score > 0:
                hits.append(SearchHit(c, score, matched_w / total_w))

        hits.sort(key=lambda h: h.score, reverse=True)
        return [h for h in hits if h.coverage >= min_coverage][:top_k]

    def summary(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c.doc, {"doc": c.doc, "title": c.title, "chunks": 0, "headings": []})
            d["chunks"] += 1
            d["headings"].append(c.heading)
        return list(docs.values())


def build_context(hits: list[SearchHit]) -> str:
    if not hits:
        return "(ไม่พบข้อมูลที่เกี่ยวข้องในคลังความรู้)"
    return "\n\n---\n\n".join(h.chunk.as_context() for h in hits)
