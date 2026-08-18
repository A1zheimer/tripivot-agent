from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable

from .models import Language

_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_MYANMAR_RE = re.compile(r"[\u1000-\u109f\uaa60-\uaa7f\ua9e0-\ua9ff]")

# High-precision indicators used by common Zawgyi detectors. This intentionally
# prefers false negatives to corrupting valid Unicode text.
_ZAWGYI_INDICATORS = (
    re.compile(r"[\u102b-\u1030\u1032]\u1039"),
    re.compile(r"\u1039[^\u1000-\u1021]"),
    re.compile(r"[\u105a\u1060-\u1097]"),
)


def normalize_text(text: str) -> str:
    text = html.unescape(text.replace("\ufeff", ""))
    text = _CONTROL_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return "\n".join(_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()).strip()


def looks_like_zawgyi(text: str) -> bool:
    return any(pattern.search(text) for pattern in _ZAWGYI_INDICATORS)


def normalize_myanmar(
    text: str,
    converter: Callable[[str], str] | None = None,
    *,
    reject_unconverted_zawgyi: bool = True,
) -> str:
    normalized = normalize_text(text)
    if not looks_like_zawgyi(normalized):
        return normalized
    if converter is not None:
        converted = normalize_text(converter(normalized))
        if not looks_like_zawgyi(converted):
            return converted
    if reject_unconverted_zawgyi:
        raise ValueError("probable Zawgyi text requires a configured converter")
    return normalized


def chinese_char_count(text: str) -> int:
    return sum(bool(_CJK_RE.fullmatch(character)) for character in text)


def pack_passages(
    text: str,
    *,
    min_cjk: int = 100,
    max_cjk: int = 480,
) -> list[str]:
    """Merge/split prose so each item meets the internship length rule."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    packed: list[str] = []
    buffer: list[str] = []
    buffer_count = 0

    def flush() -> None:
        nonlocal buffer, buffer_count
        if buffer_count >= min_cjk:
            packed.append("".join(buffer))
        buffer = []
        buffer_count = 0

    for paragraph in paragraphs:
        if _is_boilerplate(paragraph):
            continue
        count = chinese_char_count(paragraph)
        if count >= min_cjk:
            flush()
            if count <= max_cjk:
                packed.append(paragraph)
            else:
                packed.extend(_split_long_passage(paragraph, min_cjk=min_cjk, max_cjk=max_cjk))
            continue
        if buffer_count and buffer_count + count > max_cjk:
            flush()
        buffer.append(paragraph)
        buffer_count += count
        if buffer_count >= min_cjk:
            flush()
    flush()
    return packed


def _is_boilerplate(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("参见", "参考资料", "参考文献", "外部链接", "注释", "脚注")):
        return True
    if "维基百科" in stripped and chinese_char_count(stripped) < 80:
        return True
    return False


def _split_long_passage(text: str, *, min_cjk: int, max_cjk: int) -> list[str]:
    sentences = split_sentences(text, Language.CHINESE)
    packed: list[str] = []
    buffer: list[str] = []
    buffer_count = 0
    for sentence in sentences:
        count = chinese_char_count(sentence)
        if buffer_count and buffer_count + count > max_cjk:
            if buffer_count >= min_cjk:
                packed.append("".join(buffer))
            buffer = []
            buffer_count = 0
        buffer.append(sentence)
        buffer_count += count
    if buffer_count >= min_cjk:
        packed.append("".join(buffer))
    return packed


def script_ratio(text: str, language: Language) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    pattern = {
        Language.ENGLISH: _LATIN_RE,
        Language.CHINESE: _CJK_RE,
        Language.BURMESE: _MYANMAR_RE,
    }[language]
    return sum(bool(pattern.fullmatch(character)) for character in visible) / len(visible)


def is_clean_parallel_pair(
    source: str,
    target: str,
    source_language: Language,
    target_language: Language,
    *,
    min_chars: int = 2,
    max_chars: int = 600,
    max_length_ratio: float = 8.0,
) -> tuple[bool, str]:
    if not source or not target:
        return False, "empty"
    if source == target:
        return False, "identical"
    if not (min_chars <= len(source) <= max_chars and min_chars <= len(target) <= max_chars):
        return False, "length"
    if max(len(source), len(target)) / max(1, min(len(source), len(target))) > max_length_ratio:
        return False, "length_ratio"
    if _HTML_TAG_RE.search(source) or _HTML_TAG_RE.search(target):
        return False, "html"
    if source_language is Language.BURMESE and looks_like_zawgyi(source):
        return False, "zawgyi"
    if target_language is Language.BURMESE and looks_like_zawgyi(target):
        return False, "zawgyi"
    minimum_ratio = {
        Language.ENGLISH: 0.45,
        Language.CHINESE: 0.20,
        Language.BURMESE: 0.35,
    }
    if script_ratio(source, source_language) < minimum_ratio[source_language]:
        return False, "source_script"
    if script_ratio(target, target_language) < minimum_ratio[target_language]:
        return False, "target_script"
    return True, "accepted"


def split_sentences(text: str, language: Language) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    terminators = "။!?" if language is Language.BURMESE else "。！？!?"
    pattern = re.compile(rf"(?<=[{re.escape(terminators)}])\s*|\n+")
    parts = [part.strip() for part in pattern.split(normalized) if part.strip()]
    return parts or [normalized]


def chunk_sentences(sentences: Iterable[str], target_size: int = 6) -> list[str]:
    if target_size < 1:
        raise ValueError("target_size must be positive")
    items = list(sentences)
    return [
        " ".join(items[index : index + target_size])
        for index in range(0, len(items), target_size)
    ]


def term_occurs(text: str, term: str, language: Language) -> bool:
    """Language-aware term matching for English: case-insensitive and
    boundary-aware via lookarounds (so ``AI`` no longer matches ``said``).
    Unlike ``\\b``, lookarounds also match terms whose edges are symbols
    (``C++``, ``.NET``, ``C#``). Other languages use exact substring."""
    if not text or not term:
        return False
    if language is Language.ENGLISH:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        return pattern.search(text) is not None
    return term in text


# Terms must not start or end with a function character: fragments such as
# 「的计算机」 or 「可以通过」 are mining artifacts, not terminology.
TERM_BOUNDARY_STOPS = frozenset("的了是在和与或及而对于将被把有其这那不也等之并地又更很都还就个年")


def has_clean_term_boundaries(term: str) -> bool:
    return bool(term) and term[0] not in TERM_BOUNDARY_STOPS and term[-1] not in TERM_BOUNDARY_STOPS


def is_degenerate_text(text: str) -> bool:
    """Detect degenerate repetition loops (a classic small-model failure).

    Two tiers keep legitimate enumerations intact: model loops repeat the
    same span many times (>=8 for 20-char spans) or repeat long spans
    (>=60 chars, 3+ times) that prose essentially never duplicates verbatim.
    A list naming four faculties repeats a 20-char span only ~4 times and
    stays; a stuck decoder repeating one phrase 20 times goes.
    """
    if len(text) < 60:
        return False

    def top_count(window: int) -> int:
        spans = Counter(
            text[index : index + window] for index in range(0, len(text) - window + 1)
        )
        return max((count for span, count in spans.items() if span.strip()), default=0)

    return top_count(20) >= 8 or top_count(60) >= 3


# Frequent traditional-form characters used only as a data-quality indicator;
# not a converter.
_TRADITIONAL_CHARS = frozenset(
    "學國會來時間說話東車馬鳥龍風飛雲電腦網際員務發經濟濟術語資訊訊體機藝醫陽陰曆書畫"
    "國際條約組織關係聯合國務院處灣島嶼區縣鄉鎮場館廠礦鹽鐵銀圓貨幣稅賦帳貸款匯兌證券"
    "險賠償設備維修檢測試驗證書據編號碼質標準規範圍籌劃統計圖表冊頁檔案錄"
)


def traditional_char_ratio(text: str) -> float:
    cjk = [character for character in text if _CJK_RE.fullmatch(character)]
    if not cjk:
        return 0.0
    return sum(character in _TRADITIONAL_CHARS for character in cjk) / len(cjk)
