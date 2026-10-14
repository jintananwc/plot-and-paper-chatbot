"""ค้นหาหนังสือจากฐานข้อมูลของร้าน (knowledge/books.csv) แบบมีโครงสร้าง

ทำไมไม่ใช้แค่ RAG แบบค้นข้อความ: คำถามเรื่องหนังสือมักมี "เงื่อนไข" เช่น หมวด, สถานะ Available,
ค่าเช่าไม่เกิน 50 บาท, ภาษา, แพงที่สุด — ถ้าปล่อยให้โมเดลเดาจากข้อความบางส่วน มีโอกาสตอบผิด/ตกหล่น
จึงแยกเงื่อนไขจากคำถามแล้วกรองข้อมูลจริงด้วยโค้ด ส่วนคำถามเชิงความหมาย (เช่น "หนังสือเกี่ยวกับ AI")
ใช้ BM25 ค้นจาก ชื่อ/หมวด/คำอธิบาย/keywords ของหนังสือ

ผลลัพธ์ (CatalogResult) ประกอบด้วย
- books: หนังสือที่เกี่ยวข้อง (ส่งให้โมเดลเป็นบริบท)
- facts: ข้อเท็จจริงที่ระบบคำนวณให้ เช่น "ไม่พบหนังสือภาษาไทย", "ค่าปรับ 2 เล่ม × 4 วัน = 80 บาท"
"""

import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from services.rag import Chunk, KnowledgeBase

log = logging.getLogger("chatbot")

LATE_FEE_PER_BOOK_PER_DAY = 10
MAX_BOOKS_PER_RENTAL = 3

STATUS_TH = {"Available": "ว่าง พร้อมให้เช่า", "Rented": "ถูกเช่าอยู่ (จองคิวได้)", "Reserved": "มีลูกค้าจองแล้ว"}


def tier_rental_price(retail: int) -> int:
    """อัตราค่าเช่า 7 วันตามราคาปก (ตามตารางของร้าน)"""
    if retail <= 350:
        return 40
    if retail <= 450:
        return 50
    if retail <= 550:
        return 60
    return 70


@dataclass
class Book:
    book_id: str
    title: str
    author: str
    category: str
    category_th: str
    language: str
    retail_price: int
    rental_price: int
    rental_days: int
    status: str
    description_th: str
    keywords: str

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id, "title": self.title, "author": self.author,
            "category": self.category, "category_th": self.category_th, "language": self.language,
            "retail_price_thb": self.retail_price, "rental_price_7days_thb": self.rental_price,
            "rental_period_days": self.rental_days, "rental_status": self.status,
        }

    def as_context(self) -> str:
        return (f"[{self.book_id}] {self.title} — {self.author} | หมวด {self.category} ({self.category_th}) | "
                f"ภาษา {self.language} | ราคาปก {self.retail_price} บาท | ค่าเช่า {self.rental_days} วัน "
                f"{self.rental_price} บาท | สถานะ {self.status} ({STATUS_TH.get(self.status, self.status)}) | "
                f"เรื่องย่อ: {self.description_th}")


@dataclass
class CatalogResult:
    books: list[Book] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    matched_titles: list[str] = field(default_factory=list)

    def context(self) -> str:
        parts = []
        if self.facts:
            parts.append("ข้อเท็จจริงจากระบบ (คำนวณ/กรองจากฐานข้อมูลจริงแล้ว ให้ใช้ตามนี้):\n" +
                         "\n".join(f"- {f}" for f in self.facts))
        if self.books:
            parts.append("หนังสือที่เกี่ยวข้องจากฐานข้อมูลของร้าน:\n" + "\n".join(b.as_context() for b in self.books))
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# การแยกเงื่อนไขจากคำถาม
# ---------------------------------------------------------------------------

def _loose(text: str) -> str:
    """ตัวพิมพ์เล็ก, แทนเครื่องหมายด้วยช่องว่าง — ใช้เทียบชื่อหนังสือ"""
    text = re.sub(r"[^\w฀-๿]+", " ", text.casefold())
    return f" {re.sub(r'\s+', ' ', text).strip()} "


# คำเรียกหมวด (ภาษาไทย/อังกฤษ) -> category ใน CSV  (คำที่ยาวกว่าจะถูกเลือกก่อน)
CATEGORY_ALIASES: dict[str, list[str]] = {
    "Self-Development": ["self-development", "self development", "พัฒนาตนเอง", "พัฒนาตัวเอง"],
    "Productivity": ["productivity", "เพิ่มประสิทธิภาพ", "การทำงาน", "จัดการเวลา", "ประสิทธิภาพ"],
    "Finance": ["finance", "การเงิน", "การลงทุน", "ลงทุน"],
    "Business": ["business", "ธุรกิจ", "ผู้ประกอบการ", "สตาร์ทอัพ", "startup"],
    "Psychology": ["psychology", "จิตวิทยา"],
    "Fiction": ["fiction", "novel", "นิยาย", "วรรณกรรม", "นวนิยาย"],
    "Mystery & Thriller": ["mystery", "thriller", "นิยายสืบสวน", "สืบสวน", "ระทึกขวัญ"],
    "Technology & AI": ["technology", "tech", "ai", "เทคโนโลยี", "ปัญญาประดิษฐ์", "เอไอ"],
}

STATUS_ALIASES = {
    # ระบุสถานะเฉพาะคำที่ชัดเจน ("มีคนจองต่อ" ในคำถามเรื่องต่ออายุ ไม่ใช่การค้นหาหนังสือที่ถูกจอง)
    "Available": [r"(?<![a-z])available(?![a-z])", r"(?<!ระห)ว่าง", r"พร้อมให้เช่า", r"พร้อมเช่า"],
    "Rented": [r"(?<![a-z])rented(?![a-z])", r"ถูกเช่าอยู่", r"สถานะถูกเช่า"],
    "Reserved": [r"(?<![a-z])reserved(?![a-z])", r"ถูกจองแล้ว", r"สถานะถูกจอง"],
}

THAI_NUMBERS = {"หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10}
_NUM = r"(\d+|" + "|".join(THAI_NUMBERS) + r")"


def _to_int(s: str) -> int:
    return int(s) if s.isdigit() else THAI_NUMBERS[s]


def _alias_regex(alias: str) -> str:
    esc = re.escape(alias)
    return rf"(?<![a-z]){esc}(?![a-z])" if re.search(r"[a-z]", alias) else esc


def detect_categories(text: str) -> list[str]:
    """คืนรายการ category ที่ถูกพูดถึง (ตัดคำที่ซ้อนอยู่ในคำที่ยาวกว่า เช่น 'นิยาย' ใน 'นิยายสืบสวน')"""
    t = text.casefold()
    found: list[tuple[int, int, str]] = []
    for cat, aliases in CATEGORY_ALIASES.items():
        for a in aliases:
            for m in re.finditer(_alias_regex(a), t):
                found.append((m.start(), m.end(), cat))
    kept = [f for f in found if not any(o[0] <= f[0] and f[1] <= o[1] and (o[1] - o[0]) > (f[1] - f[0]) for o in found)]
    out: list[str] = []
    for _, _, cat in sorted(kept):
        if cat not in out:
            out.append(cat)
    return out


def detect_status(text: str) -> str | None:
    t = text.casefold()
    for status, pats in STATUS_ALIASES.items():
        if any(re.search(p, t) for p in pats):
            return status
    return None


def detect_language(text: str) -> str | None:
    t = text.casefold()
    if re.search(r"ภาษาไทย|(?<![a-z])thai(?![a-z])|ฉบับไทย|แปลไทย", t):
        return "Thai"
    if re.search(r"ภาษาอังกฤษ|(?<![a-z])english(?![a-z])|ฉบับอังกฤษ", t):
        return "English"
    return None


def detect_price_filter(text: str) -> dict | None:
    """'ค่าเช่าไม่เกิน 50 บาท' -> {"field": "rental", "op": "<=", "value": 50}"""
    t = text.casefold().replace(",", "")
    field_name = "retail" if re.search(r"ราคาปก|ราคาหนังสือ|ราคาเต็ม|ราคาเล่ม|retail", t) else "rental"
    rules = [
        (r"(ไม่เกิน|ไม่ถึง|ไม่เกินกว่า|ต่ำกว่าหรือเท่ากับ|<=|at most|under|below|งบ)\s*" + r"(\d+)", "<="),
        (r"(ต่ำกว่า|น้อยกว่า|ถูกกว่า|less than|<)\s*(\d+)", "<"),
        (r"(มากกว่า|สูงกว่า|แพงกว่า|เกินกว่า|more than|over|>)\s*(\d+)", ">"),
        (r"(ตั้งแต่|อย่างน้อย|at least|>=)\s*(\d+)", ">="),
    ]
    for pat, op in rules:
        m = re.search(pat, t)
        if m:
            value = int(m.group(2))
            if op == "<" and m.group(1) in ("ไม่ถึง",):
                op = "<"
            return {"field": field_name, "op": op, "value": value}
    # "เกิน 50" แบบไม่มี "ไม่" นำหน้า
    m = re.search(r"(?<!ไม่)เกิน\s*(\d+)\s*บาท", t)
    if m:
        return {"field": field_name, "op": ">", "value": int(m.group(1))}
    return None


def detect_superlative(text: str) -> str | None:
    t = text.casefold()
    if re.search(r"แพงที่สุด|แพงสุด|ราคาสูงที่สุด|ราคาสูงสุด|most expensive|highest price", t):
        return "max"
    if re.search(r"ถูกที่สุด|ถูกสุด|ราคาต่ำที่สุด|ราคาต่ำสุด|cheapest|lowest price", t):
        return "min"
    return None


def late_fee_fact(text: str) -> str | None:
    """คำนวณค่าปรับคืนช้า ถ้าคำถามระบุจำนวนวันที่คืนช้า"""
    t = text.casefold()
    if not re.search(r"ช้า|เกินกำหนด|late|ค่าปรับ|overdue", t):
        return None
    m_days = re.search(r"(?:ช้า|เกินกำหนด|เกิน|late|overdue)\D{0,12}?" + _NUM + r"\s*(?:วัน|days?)", t) or \
        re.search(_NUM + r"\s*(?:วัน|days?)\D{0,8}?(?:ช้า|late|เกินกำหนด)", t)
    if not m_days:
        return None
    days = _to_int(m_days.group(1))
    m_books = re.search(_NUM + r"\s*(?:เล่ม|books?)", t)
    books = _to_int(m_books.group(1)) if m_books else 1
    fee = books * days * LATE_FEE_PER_BOOK_PER_DAY
    fact = (f"ค่าปรับคืนช้า (Late Return Policy): {books} เล่ม × {days} วัน × {LATE_FEE_PER_BOOK_PER_DAY} บาท "
            f"= {fee} บาท")
    if books > MAX_BOOKS_PER_RENTAL:
        fact += f" (หมายเหตุ: ปกติเช่าได้สูงสุด {MAX_BOOKS_PER_RENTAL} เล่มต่อครั้ง)"
    return fact


# ---------------------------------------------------------------------------
# ฐานข้อมูลหนังสือ
# ---------------------------------------------------------------------------

BOOK_SEARCH_STOPWORDS = [
    "หนังสือ", "ให้เช่า", "ค่าเช่า", "เช่า", "เล่มไหน", "เล่ม", "แนะนำ", "เกี่ยวกับ", "อยากได้", "อยากอ่าน",
    "หมวด", "ร้าน", "มีไหม", "มี", "แนว", "เรื่อง", "ตอนนี้", "บ้าง", "หน่อย",
]


class Catalog:
    def __init__(self, csv_path: str):
        self.path = Path(csv_path)
        self.books: list[Book] = []
        self.warnings: list[str] = []
        self._index: KnowledgeBase | None = None
        self.load()

    def load(self) -> None:
        books, warnings = [], []
        if not self.path.exists():
            log.warning("ไม่พบไฟล์ฐานข้อมูลหนังสือ: %s", self.path)
        else:
            with open(self.path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        b = Book(
                            book_id=row["book_id"].strip(), title=row["title"].strip(), author=row["author"].strip(),
                            category=row["category"].strip(), category_th=row["category_th"].strip(),
                            language=row["language"].strip(), retail_price=int(float(row["retail_price_thb"])),
                            rental_price=int(float(row["rental_price_7days_thb"])),
                            rental_days=int(float(row["rental_period_days"] or 7)), status=row["rental_status"].strip(),
                            description_th=row.get("description_th", "").strip(), keywords=row.get("keywords", "").strip(),
                        )
                    except (KeyError, ValueError) as exc:
                        warnings.append(f"ข้ามแถวที่อ่านไม่ได้: {row.get('book_id', '?')} ({exc})")
                        continue
                    if tier_rental_price(b.retail_price) != b.rental_price:
                        warnings.append(f"{b.book_id} ค่าเช่า {b.rental_price} ไม่ตรงตารางราคา "
                                        f"(ราคาปก {b.retail_price} ควรเป็น {tier_rental_price(b.retail_price)})")
                    if b.status not in STATUS_TH:
                        warnings.append(f"{b.book_id} สถานะไม่รู้จัก: {b.status}")
                    books.append(b)
        self.books, self.warnings = books, warnings
        for w in warnings:
            log.warning(w)
        self._build_index()

    def _build_index(self) -> None:
        idx = KnowledgeBase.__new__(KnowledgeBase)
        idx.k1, idx.b = 1.5, 0.75
        idx.directory = self.path
        chunks = [Chunk(b.book_id, "books.csv", b.title, b.category,
                        f"{b.title} {b.author} {b.category} {b.category_th} {b.description_th} {b.keywords}")
                  for b in self.books]
        idx.set_chunks(chunks)
        self._index = idx

    def by_id(self, book_id: str) -> Book | None:
        return next((b for b in self.books if b.book_id == book_id), None)

    def overview(self) -> str:
        status = Counter(b.status for b in self.books)
        cats = Counter(f"{b.category} ({b.category_th})" for b in self.books)
        langs = Counter(b.language for b in self.books)
        return (f"ภาพรวมฐานข้อมูลหนังสือ: ทั้งหมด {len(self.books)} เรื่อง | สถานะ " +
                ", ".join(f"{k} {v}" for k, v in status.items()) + " | หมวด " +
                ", ".join(f"{k} {v} เรื่อง" for k, v in cats.items()) + " | ภาษา " +
                ", ".join(f"{k} {v} เรื่อง" for k, v in langs.items()))

    def title_index(self) -> str:
        """รายชื่อหนังสือทั้งหมดแบบย่อ (1 บรรทัดต่อเล่ม) — ส่งให้โมเดลทุกครั้ง
        เพื่อให้รู้เสมอว่าร้านมี/ไม่มีเล่มไหน แม้รอบนั้นจะค้นเจอแค่บางเล่ม"""
        return "\n".join(f"[{b.book_id}] {b.title} — {b.author} | {b.category} | ค่าเช่า {b.rental_price} บาท | {b.status}"
                         for b in self.books)

    # ---------- ค้นหา ----------
    def match_titles(self, text: str) -> tuple[list[Book], str]:
        """หาชื่อหนังสือที่ถูกพูดถึง คืน (หนังสือ, ข้อความที่ตัดชื่อออกแล้ว)"""
        q = _loose(text)
        found = []
        for b in sorted(self.books, key=lambda x: -len(x.title)):
            t = _loose(b.title)
            variants = {t, " " + t.strip().removeprefix("the ").strip() + " "}
            for v in variants:
                if len(v.strip()) >= 3 and v in q:
                    found.append(b)
                    q = q.replace(v, " ")
                    break
        return found, q

    def filter(self, *, categories=None, status=None, language=None, price=None) -> list[Book]:
        out = list(self.books)
        if categories:
            out = [b for b in out if b.category in categories]
        if status:
            out = [b for b in out if b.status == status]
        if language:
            out = [b for b in out if b.language.casefold() == language.casefold()]
        if price:
            val = (lambda b: b.retail_price) if price["field"] == "retail" else (lambda b: b.rental_price)
            ops = {"<=": lambda x: x <= price["value"], "<": lambda x: x < price["value"],
                   ">": lambda x: x > price["value"], ">=": lambda x: x >= price["value"]}
            out = [b for b in out if ops[price["op"]](val(b))]
        return out

    def semantic(self, text: str, top_k: int = 5, min_coverage: float = 0.35) -> list[Book]:
        if not self._index:
            return []
        hits = self._index.search(text, top_k, min_coverage, extra_stopwords=BOOK_SEARCH_STOPWORDS)
        return [b for h in hits if (b := self.by_id(h.chunk.id))]

    def search(self, text: str, max_books: int = 20) -> CatalogResult:
        res = CatalogResult()
        if fee := late_fee_fact(text):
            res.facts.append(fee)

        titled, rest = self.match_titles(text)
        res.matched_titles = [b.title for b in titled]
        categories = detect_categories(rest)
        status = detect_status(rest)
        language = detect_language(rest)
        price = detect_price_filter(rest)
        superlative = detect_superlative(rest)
        res.filters = {k: v for k, v in {"categories": categories, "status": status, "language": language,
                                          "price": price, "superlative": superlative}.items() if v}

        books: list[Book] = list(titled)
        if len(titled) >= 2 and re.search(r"รวม|ทั้งหมด|total|ทั้งสอง|ทั้งสาม", text.casefold()):
            total = sum(b.rental_price for b in titled)
            res.facts.append("ค่าเช่ารวม 7 วัน: " + " + ".join(f"{b.title} {b.rental_price}" for b in titled) +
                             f" = {total} บาท")
        if len(titled) > MAX_BOOKS_PER_RENTAL:
            res.facts.append(f"ลูกค้าระบุ {len(titled)} เล่ม แต่เช่าได้สูงสุด {MAX_BOOKS_PER_RENTAL} เล่มต่อครั้ง")

        if superlative:
            pool = self.filter(categories=categories, status=status, language=language)
            if pool:
                pick = max if superlative == "max" else min
                word = "สูงที่สุด" if superlative == "max" else "ต่ำที่สุด"
                best_retail = pick(b.retail_price for b in pool)
                best_rent = pick(b.rental_price for b in pool)
                r_books = [b for b in pool if b.retail_price == best_retail]
                t_books = [b for b in pool if b.rental_price == best_rent]
                res.facts.append(f"หนังสือที่ราคาปก{word}: " + ", ".join(b.title for b in r_books) +
                                 f" (ราคาปก {best_retail} บาท)")
                res.facts.append(f"ค่าเช่า 7 วัน{word} คือ {best_rent} บาท: " + ", ".join(b.title for b in t_books))
                for b in r_books + t_books:
                    if b not in books:
                        books.append(b)

        # ถามถึงเล่มที่ระบุชื่อ ("Rich Dad Poor Dad ว่างไหม") -> คำว่า "ว่าง" คือถามสถานะของเล่มนั้น ไม่ใช่ให้กรองทั้งร้าน
        has_filter = bool(categories or price or ((status or language) and not titled))
        if has_filter and not superlative:
            matched = self.filter(categories=categories, status=status, language=language, price=price)
            desc = self._describe_filters(categories, status, language, price)
            if matched:
                res.facts.append(f"ผลการกรองตามเงื่อนไข ({desc}): พบ {len(matched)} เรื่อง — " +
                                 ", ".join(f"{b.title} ({b.status}, ค่าเช่า {b.rental_price} บาท)" for b in matched))
            else:
                res.facts.append(f"ไม่พบหนังสือที่ตรงเงื่อนไข ({desc}) ในฐานข้อมูลของร้าน")
            books += [b for b in matched if b not in books]

        if not titled and not has_filter and not superlative:
            sem = self.semantic(rest)
            books += [b for b in sem if b not in books]

        res.books = books[:max_books]
        return res

    @staticmethod
    def _describe_filters(categories, status, language, price) -> str:
        parts = []
        if categories:
            parts.append("หมวด " + " / ".join(categories))
        if status:
            parts.append(f"สถานะ {status}")
        if language:
            parts.append(f"ภาษา {'ไทย' if language == 'Thai' else 'อังกฤษ'}")
        if price:
            name = "ราคาปก" if price["field"] == "retail" else "ค่าเช่า"
            op = {"<=": "ไม่เกิน", "<": "ต่ำกว่า", ">": "มากกว่า", ">=": "ตั้งแต่"}[price["op"]]
            parts.append(f"{name}{op} {price['value']} บาท")
        return ", ".join(parts)
