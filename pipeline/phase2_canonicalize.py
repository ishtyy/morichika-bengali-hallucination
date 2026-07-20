"""Pure label-free text canonicalization shared by Phase 2 runtime stages."""

from __future__ import annotations

import re
import unicodedata


VERSION = "phase2-symbol-safe-canonicalizer-v2-idempotent-boundaries"
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def canonicalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).translate(BN_DIGITS)
    text = text.casefold().replace("&gt;", ">").replace("&lt;", "<")
    text = re.sub(r"[*]+", "", text)
    text = re.sub(r"[“”\"'`‘’]+", "", text)
    text = re.sub(r"[^\w\u0980-\u09ff%√<>/=+.\-@$#&]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Strip boundary punctuation and the whitespace adjacent to it in one
    # operation.  The v1 ``strip().strip('.-')`` sequence could expose a new
    # leading/trailing space, so canonicalize(canonicalize(x)) differed from
    # canonicalize(x) for OCR strings such as ``".. question"``.
    return re.sub(r"^[.\-\s]+|[.\-\s]+$", "", text)
