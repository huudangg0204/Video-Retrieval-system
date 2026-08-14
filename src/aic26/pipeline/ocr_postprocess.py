"""
OCR Post-processing pipeline cho dữ liệu tin tức tiếng Việt.

Sửa các lỗi phổ biến từ PaddleOCR/EasyOCR khi xử lý frame video
đài truyền hình Việt Nam (HTV, HTV7, HTV9):

    1. Logo đài truyền hình bị nhận sai  (HTY → HTV9)
    2. Định dạng thời gian bị lệch       (0630:22 → 06:30:22)
    3. Nhầm lẫn chữ cái ↔ chữ số         (2O25 → 2025)
    4. Ký tự tiếng Việt bị ép sang Latin  (Qu�c → Quốc)

Mỗi bước là một hàm thuần (pure function) nhận str → str,
pipeline ghép tuần tự qua ``clean_ocr_text()``.

Usage::

    from aic26.pipeline.ocr_postprocess import run_postprocess_pipeline
    run_postprocess_pipeline("data/ocr_results.csv", "data/ocr_results_clean.csv")

References
----------
- PEP 8 compliant, type-annotated.
- Designed for ``pandas.Series.apply`` (vectorisable where possible).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# Fix Windows console encoding for Vietnamese output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════════════
#  1. SỬA LỖI LOGO ĐÀI TRUYỀN HÌNH
# ════════════════════════════════════════════════════════════════════════

# Regex bắt các biến thể sai của "HTV", "HTV7", "HTV9" ở ĐẦU chuỗi.
#
# Phân tích từ frequency data thực tế trên 41K+ frames:
#
# ┌─────────── Prefix ────────────┐  ┌── Suffix ──┐
# H + (T|7|/) + (V|Y|y|O|S|D|L|C|R|I|i|/) + (V|O|S|g)? + (7|9|79|77)?
#
# Variants bắt được:
#   HTY, HTYO, HTVO, HTyO, HTVS, HTYS, HTyS, Hty9, HtyS, Htys, HTyg
#   HTVI, HTYi, HTV7O, HTV7S, HTV79, HTV77, H7T, H77, HT7, HT79
#   HT/V, HT/I, HT/D, HT/, HTD, HTL, HTVC, HTVR, HT (alone), HT9
_LOGO_RE = re.compile(
    r"""
    ^                       # anchor đầu chuỗi
    (?:                     # --- nhóm không bắt: toàn bộ logo ---
        [Hh]               # H hoặc h
        [Tt7/]             # T, 7, hoặc /
        (?:                # phần thân logo (tuỳ chọn cho trường hợp "HT")
            [/]?           # dấu / tuỳ chọn (HT/V, HT/I)
            [VvYyOoSsDdLlCcRrIiUugq]  # ký tự bị nhầm
            [VvOoSsIi]?    # tuỳ chọn thêm 1 ký tự (HTVO, HTYS, HTVI)
        )?
    )
    (?:                     # --- nhóm không bắt: hậu tố kênh ---
        7{1,2}             # "7" hoặc "77" → normalise thành "7"
        [OoSs]?            # HTV7O, HTV7S → HTV7
      | 9                  # "9"
    )?
    (?=\s|$)               # look-ahead: theo sau là khoảng trắng hoặc cuối
    """,
    re.VERBOSE,
)

# Mapping hậu tố: nhìn ký tự cuối của match để quyết định kênh
_CHANNEL_SUFFIX_MAP = {"7": "7", "9": "9"}


def fix_tv_logo(text: str) -> str:
    """Chuẩn hoá logo đài truyền hình ở đầu chuỗi OCR.

    Rules
    -----
    - Bất kỳ biến thể nào khớp ``_LOGO_RE`` đều được thay bằng
      ``HTV``, ``HTV7``, hoặc ``HTV9`` tuỳ hậu tố.
    - Nếu không khớp → trả nguyên.

    Examples
    --------
    >>> fix_tv_logo("HTY 06:30:22 giay")
    'HTV9 06:30:22 giay'
    >>> fix_tv_logo("HTVO 06:45:55 giay")
    'HTV9 06:45:55 giay'
    >>> fix_tv_logo("HTV7 18:33:41 tin tuc")
    'HTV7 18:33:41 tin tuc'
    """

    def _replace(m: re.Match) -> str:
        matched = m.group(0)
        # Xác định hậu tố kênh từ ký tự cuối
        last_char = matched[-1]
        suffix = _CHANNEL_SUFFIX_MAP.get(last_char, "")
        # Nếu không có suffix rõ ràng (HTY, HTVO, HTVS...) → mặc định HTV9
        # vì phần lớn lỗi đến từ HTV9 bị OCR sai
        if not suffix:
            suffix = "9"
        return f"HTV{suffix}"

    return _LOGO_RE.sub(_replace, text, count=1)


# ════════════════════════════════════════════════════════════════════════
#  2. SỬA LỖI ĐỊNH DẠNG THỜI GIAN
# ════════════════════════════════════════════════════════════════════════

# Regex bắt các chuỗi "giống thời gian" nhưng bị lệch format.
# Pattern 1: HHMM:SS  (thiếu dấu : giữa HH và MM)    VD: 0630:22
# Pattern 2: HH:MMSS  (thiếu dấu : giữa MM và SS)    VD: 06:3022
# Pattern 3: HHMMSS   (thiếu cả hai dấu :)            VD: 063022
# Pattern 4: HH:MM:SS nhưng có chữ O thay 0           VD: O6:30:22
#
# Chỉ khớp chuỗi 6-8 ký tự gồm digits, 'O', và ':'.
_TIME_MALFORMED_RE = re.compile(
    r"""
    (?<!\d)                  # không đứng sau digit (tránh khớp giữa số dài)
    (                        # --- capture group ---
        [O\d]{2,4}           #   2-4 ký tự (HH hoặc HHMM)
        :                    #   dấu hai chấm
        [O\d]{2,6}           #   2-6 ký tự (MM, MMSS, hoặc SS)
        (?::[O\d]{2})?       #   tuỳ chọn: :SS
    )
    (?!\d)                   # không theo sau digit
    """,
    re.VERBOSE,
)

# Regex bắt chuỗi 6 chữ số liền nhau có thể là HHMMSS
_TIME_NO_COLON_RE = re.compile(
    r"(?<!\d)([O\d]{6})(?!\d)"
)


def _o_to_zero(s: str) -> str:
    """Thay chữ 'O' (in hoa) bằng số '0' trong chuỗi thời gian."""
    return s.replace("O", "0")


def fix_time_format(text: str) -> str:
    """Chuẩn hoá chuỗi thời gian về dạng HH:MM:SS.

    Xử lý tuần tự:
    1. Thay O → 0 trong mọi cụm giống thời gian.
    2. Chèn dấu ``:`` vào đúng vị trí nếu thiếu.
    3. Bỏ qua chuỗi không hợp lệ (giờ > 23, phút/giây > 59).

    Examples
    --------
    >>> fix_time_format("0630:22")
    '06:30:22'
    >>> fix_time_format("O6:30:35")
    '06:30:35'
    >>> fix_time_format("1853:08")
    '18:53:08'
    >>> fix_time_format("06:3529")
    '06:35:29'
    """

    def _fix_time_match(m: re.Match) -> str:
        raw = _o_to_zero(m.group(1))
        # Loại bỏ tất cả dấu : để lấy digits thuần
        digits = raw.replace(":", "")

        # Chỉ xử lý nếu có đúng 6 chữ số (HHMMSS)
        if len(digits) == 6 and digits.isdigit():
            hh, mm, ss = digits[:2], digits[2:4], digits[4:6]
            if int(hh) <= 23 and int(mm) <= 59 and int(ss) <= 59:
                return f"{hh}:{mm}:{ss}"

        # Nếu đã ở dạng HH:MM:SS → chỉ cần fix O→0
        if re.match(r"^\d{2}:\d{2}:\d{2}$", raw):
            return raw

        # Trả nguyên nếu không parse được
        return m.group(0)

    # Bước 1: xử lý các chuỗi đã có ít nhất 1 dấu ':'
    text = _TIME_MALFORMED_RE.sub(_fix_time_match, text)

    # Bước 2: xử lý chuỗi 6 chữ số liền (HHMMSS) — ít phổ biến hơn
    def _fix_bare_digits(m: re.Match) -> str:
        raw = _o_to_zero(m.group(1))
        if raw.isdigit() and len(raw) == 6:
            hh, mm, ss = raw[:2], raw[2:4], raw[4:6]
            if int(hh) <= 23 and int(mm) <= 59 and int(ss) <= 59:
                return f"{hh}:{mm}:{ss}"
        return m.group(0)

    text = _TIME_NO_COLON_RE.sub(_fix_bare_digits, text)

    return text


# ════════════════════════════════════════════════════════════════════════
#  3. SỬA LỖI NHẦM LẪN CHỮ CÁI ↔ CHỮ SỐ
# ════════════════════════════════════════════════════════════════════════

# Pattern: chữ O (in hoa) xuất hiện giữa các chữ số → chắc chắn là 0.
# VD: 2O25 → 2025,  1O0% → 100%,  3O → 30
_DIGIT_O_RE = re.compile(
    r"(?<=\d)O(?=\d)"   # O đứng giữa 2 chữ số → thay bằng 0
)

# Pattern: O đứng ngay trước 1+ chữ số (đầu token)
# VD: O6 → 06 (trong ngữ cảnh thời gian O6:30)
_LEADING_O_RE = re.compile(
    r"(?<!\w)O(?=\d)"   # O ở đầu word, theo sau là digit
)


def fix_digit_letter_confusion(text: str) -> str:
    """Sửa nhầm lẫn O↔0 trong ngữ cảnh số.

    Examples
    --------
    >>> fix_digit_letter_confusion("2O25")
    '2025'
    >>> fix_digit_letter_confusion("nam 2O29 du kien")
    'nam 2029 du kien'
    """
    text = _DIGIT_O_RE.sub("0", text)
    text = _LEADING_O_RE.sub("0", text)
    return text


# ════════════════════════════════════════════════════════════════════════
#  4. XỬ LÝ LỖI KÝ TỰ TIẾNG VIỆT
# ════════════════════════════════════════════════════════════════════════

# OCR model (PaddleOCR/EasyOCR) thường ép tiếng Việt sang Latin:
#   - ö  → ô, ố, ồ, ộ, ổ, ỗ  (o-circumflex variants)
#   - ä  → â, ấ, ầ, ậ, ẩ, ẫ  (a-circumflex variants)
#   - ü  → ư, ứ, ừ, ự, ử, ữ  (u-hook variants)
#   - @  → ê, ế, ề, ệ, ể, ễ  (e-circumflex variants)
#   - �  → nhiều ký tự khác nhau (replacement char U+FFFD)
#
# Vì không có dấu thanh (sắc, huyền, nặng…), ta chỉ map về base char.
# Mapping xây dựng từ frequency analysis thực tế trên dataset.

# ── Word-level mapping (chính xác nhất) ──
# Các từ sai phổ biến → từ đúng tiếng Việt.
# Build từ top-frequency words trong dataset.
WORD_CORRECTIONS: dict[str, str] = {
    # --- ö → ô/ố/ồ/ộ ---
    "khöng": "không",
    "Qu\u00f6c": "Quốc",      # Quöc
    "qu\u00f6c": "quốc",
    "d\u00f6ng": "đồng",      # döng (note: đ→d cũng bị mất)
    "D\u00f6ng": "Đồng",
    "c\u00f6ng": "công",
    "C\u00f6ng": "Công",
    "du\u00f6ng": "đường",
    "th\u00f6ng": "thông",
    "ngu\u00f6i": "người",
    "tru\u00f6ng": "trường",
    "phu\u00f6ng": "phương",
    "s\u00f6ng": "sông",
    "ph\u00f6ng": "phòng",
    "H\u00f6ng": "Hồng",
    "h\u00f6ng": "hồng",
    "v\u00f6ng": "vòng",
    "d\u00f6ng": "động",
    "d\u00f6i": "đổi",
    "s\u00f6": "số",
    "m\u00f6i": "mỗi",
    "nh\u00f6": "nhỏ",
    "thu\u00f6c": "thuộc",
    "cu\u00f6c": "cuộc",
    "bu\u00f6c": "buộc",
    "tu\u00f6i": "tuổi",
    # --- ä → â/ấ/ầ/ậ ---
    "d\u00e4u": "đầu",
    "d\u00e4t": "đất",
    "n\u00e4m": "năm",
    "t\u00e4ng": "tăng",
    "nh\u00e4n": "nhân",
    "xu\u00e4t": "xuất",
    "s\u00e4n": "sản",
    "c\u00e4u": "cầu",
    "c\u00e4p": "cấp",
    "n\u00e4ng": "năng",
    "d\u00e4n": "dân",
    "nh\u00e4t": "nhất",
    "b\u00e4n": "bản",
    "b\u00e4o": "bão",
    "B\u00e4c": "Bắc",
    "b\u00e4c": "bắc",
    "ph\u00e4t": "phát",
    "th\u00e4nh": "thành",
    "h\u00e4nh": "hành",
    "tr\u00e4m": "trăm",
    "kh\u00e4n": "khẩn",
    "l\u00e4n": "lần",
    "t\u00e4ng": "tầng",
    "Ch\u00e4u": "Châu",
    "ch\u00e4u": "châu",
    # --- ü → ư/ứ/ừ ---
    "c\u00fca": "của",
    "c\u00fcu": "cứu",
    "ch\u00fcc": "chức",
    "d\u00fcng": "dừng",
    "l\u00fcc": "lực",
    "th\u00fcc": "thực",
    "ph\u00fcc": "phúc",
    "d\u00fct": "dứt",
    "Th\u00fc": "Thủ",
    "th\u00fc": "thủ",
    "t\u00fcy": "tuy",
    "r\u00fct": "rút",
    "s\u00fcc": "sức",
    "ch\u00fcng": "chứng",
    "d\u00fcng": "dũng",
    "t\u00fc": "tử",
    # --- @ → ê/ế/ề/ệ ---
    "Vi@t": "Việt",
    "vi@t": "việt",
    "l@": "lê",           # có thể là lề/lệ tuỳ ngữ cảnh
    "L@": "Lê",
    "tri@u": "triều",
    "thi@t": "thiết",
    "s@": "sẽ",
    "l@nh": "lệnh",
    "li@u": "liệu",
    "vi@c": "việc",
    "nghi@p": "nghiệp",
    "d@": "để",            # hoặc "dễ" tuỳ ngữ cảnh
    "di@n": "điện",
    "ni@m": "niềm",
    "hi@n": "hiện",
    "vi@n": "viện",
    "h@": "hệ",
    "t@": "tế",
    "k@": "kế",
    "n@n": "nền",
    "d@n": "đến",
    "TULI@U": "TƯ LIỆU",
    # --- Lỗi phổ biến khác ---
    "giay": "giây",
    "ngay": "ngày",
}

# ── Character-level fallback ──
# Khi word-level mapping không khớp, thay thế ký tự đơn lẻ.
# Đây là best-effort: mất dấu thanh nhưng đỡ hơn ký tự Latin.
_CHAR_FALLBACK: dict[str, str] = {
    "\u00f6": "ô",    # ö → ô
    "\u00e4": "â",    # ä → â
    "\u00fc": "ư",    # ü → ư
    "@": "ê",          # @ → ê (trong ngữ cảnh từ tiếng Việt)
}

# Pre-compile regex cho character fallback
# Chỉ thay @ khi nằm trong từ (có chữ cái liền kề)
_AT_IN_WORD_RE = re.compile(r"(?<=[a-zA-ZÀ-ỹ])@(?=[a-zA-ZÀ-ỹ])")
_GERMAN_CHAR_RE = re.compile(r"[\u00f6\u00e4\u00fc]")  # ö, ä, ü

# Replacement char U+FFFD — ta chỉ xoá nó vì không biết gốc là gì
_REPLACEMENT_CHAR_RE = re.compile(r"\ufffd")


def fix_vietnamese_chars(text: str) -> str:
    """Sửa ký tự tiếng Việt bị ép sang Latin/German.

    Strategy (cascading):
    1. Word-level: tra ``WORD_CORRECTIONS`` dict cho từng token.
    2. Character-level: thay ö→ô, ä→â, ü→ư, @→ê.
    3. Cleanup: xoá ký tự replacement U+FFFD.

    Examples
    --------
    >>> fix_vietnamese_chars("Han Quöc")
    'Han Quốc'
    >>> fix_vietnamese_chars("Bäc Giang")
    'Bắc Giang'
    >>> fix_vietnamese_chars("khöng d@")
    'không để'
    """
    # Bước 1: Word-level correction
    tokens = text.split()
    corrected_tokens: list[str] = []
    for token in tokens:
        # Tách punctuation trailing (VD: "Quöc," → "Quöc" + ",")
        # Giữ nguyên punctuation
        stripped = token.rstrip(".,;:!?\"'()[]{}…–—")
        trailing = token[len(stripped):]

        if stripped in WORD_CORRECTIONS:
            corrected_tokens.append(WORD_CORRECTIONS[stripped] + trailing)
        else:
            corrected_tokens.append(token)

    text = " ".join(corrected_tokens)

    # Bước 2: Character-level fallback cho ö, ä, ü
    text = _GERMAN_CHAR_RE.sub(
        lambda m: _CHAR_FALLBACK.get(m.group(0), m.group(0)), text
    )

    # Bước 3: @ → ê (chỉ khi nằm giữa chữ cái)
    text = _AT_IN_WORD_RE.sub("ê", text)

    # Bước 4: Xoá replacement character
    text = _REPLACEMENT_CHAR_RE.sub("", text)

    return text


# ════════════════════════════════════════════════════════════════════════
#  5. PIPELINE TỔNG HỢP
# ════════════════════════════════════════════════════════════════════════

def clean_ocr_text(text: str) -> str:
    """Chạy toàn bộ pipeline post-processing trên một chuỗi OCR.

    Thứ tự thực hiện:
    1. Sửa logo đài truyền hình
    2. Sửa nhầm lẫn chữ ↔ số (O↔0)
    3. Sửa định dạng thời gian
    4. Sửa ký tự tiếng Việt

    Args:
        text: Chuỗi OCR thô từ text_full hoặc texts.

    Returns:
        Chuỗi đã được dọn dẹp.
    """
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""

    text = fix_tv_logo(text)
    text = fix_digit_letter_confusion(text)
    text = fix_time_format(text)
    text = fix_vietnamese_chars(text)

    return text


def clean_texts_list(texts_raw: str | list) -> str:
    """Dọn dẹp cột ``texts`` (danh sách chuỗi, lưu dạng string repr).

    Args:
        texts_raw: Giá trị cột ``texts``, có thể là string repr
                   của list (VD: "['abc', 'def']") hoặc list thực.

    Returns:
        String repr của list đã clean.
    """
    if isinstance(texts_raw, str):
        try:
            texts_list = ast.literal_eval(texts_raw)
        except (ValueError, SyntaxError):
            return texts_raw
    elif isinstance(texts_raw, list):
        texts_list = texts_raw
    else:
        return str(texts_raw)

    cleaned = [clean_ocr_text(t) if isinstance(t, str) else t for t in texts_list]
    return str(cleaned)


# ════════════════════════════════════════════════════════════════════════
#  6. ENTRY POINT: CHẠY PIPELINE TRÊN TOÀN BỘ CSV
# ════════════════════════════════════════════════════════════════════════

def run_postprocess_pipeline(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    show_samples: int = 10,
) -> pd.DataFrame:
    """Chạy pipeline dọn dẹp OCR trên toàn bộ file CSV.

    Args:
        input_csv: Đường dẫn tới ``ocr_results.csv`` gốc.
        output_csv: Đường dẫn output. Nếu ``None``, ghi đè file gốc.
        show_samples: Số dòng mẫu hiển thị so sánh trước/sau.

    Returns:
        DataFrame đã dọn dẹp.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv) if output_csv else input_csv

    logger.info(f"Loading OCR data from {input_csv}...")
    df = pd.read_csv(input_csv)
    n_total = len(df)
    logger.info(f"Loaded {n_total} rows.")

    # Lưu bản gốc để so sánh
    df["text_full_original"] = df["text_full"]

    # ── Apply pipeline ──
    logger.info("Applying OCR post-processing pipeline...")

    # Clean text_full
    df["text_full"] = df["text_full"].fillna("").apply(clean_ocr_text)

    # Clean texts (list column)
    df["texts"] = df["texts"].fillna("[]").apply(clean_texts_list)

    # ── Thống kê thay đổi ──
    changed_mask = df["text_full"] != df["text_full_original"]
    n_changed = changed_mask.sum()
    logger.info(
        f"Post-processing complete: {n_changed}/{n_total} rows changed "
        f"({n_changed / max(n_total, 1) * 100:.1f}%)"
    )

    # ── In mẫu so sánh ──
    if show_samples > 0:
        sample_df = df[changed_mask].head(show_samples)
        print("\n" + "=" * 80)
        print("TRƯỚC vs SAU post-processing")
        print("=" * 80)
        for _, row in sample_df.iterrows():
            print(f"\n  Video: {row['video']} / {row['file']}")
            print(f"  TRƯỚC: {row['text_full_original']}")
            print(f"  SAU:   {row['text_full']}")
            print(f"  {'─' * 70}")

    # ── Xoá cột tạm ──
    df.drop(columns=["text_full_original"], inplace=True)

    # ── Ghi output ──
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Cleaned CSV saved to {output_csv}")

    return df


# ── CLI support ──
if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/ocr_results.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_postprocess_pipeline(input_path, output_path)
