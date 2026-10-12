"""The one normaliser, used on both sides of every comparison, and the matching built on it.

norm() lowers case, drops Hebrew points (niqqud and cantillation) and punctuation, turns dashes and the
maqaf into spaces, and collapses whitespace. It does not fold Hebrew spelling variants or strip prefixes:
a transcript that writes a word differently is different, and content-word recall decides whether that
matters.
"""
from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

# The script each language's spoken line must be written in, by Unicode character name.
SCRIPT = {'he': 'HEBREW', 'ar': 'ARABIC', 'ru': 'CYRILLIC', 'uk': 'CYRILLIC', 'zh': 'CJK', 'ja': 'CJK',
          'en': 'LATIN', 'es': 'LATIN', 'fr': 'LATIN', 'pt': 'LATIN', 'de': 'LATIN', 'it': 'LATIN'}


def norm(s: str | None) -> str:
    out = []
    for ch in (s or '').lower():
        cat = unicodedata.category(ch)
        if cat == 'Mn' and '֑' <= ch <= 'ׇ':
            continue
        if cat == 'Pd':
            out.append(' ')
        elif cat.startswith('P'):
            continue
        else:
            out.append(ch)
    return ' '.join(''.join(out).split())


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def title_score(expected: str, actual: str) -> float:
    """Similarity, or 1.0 when the actual title holds every word of the expected one as a whole word.

    "water the beds tonight" is a right title for "water the beds". The reverse is not: "seeds" for
    "tomato seeds" lost a word."""
    want, got = norm(expected).split(), set(norm(actual).split())
    return 1.0 if want and all(w in got for w in want) else similar(expected, actual)


def best_ratio(expected, actual: str) -> float:
    """The best title_score over the accepted forms (a string, or a list of strings)."""
    return max(title_score(e, actual) for e in (expected if isinstance(expected, list) else [expected]))


def word_present(word, hyp_norm: str) -> bool:
    """A content word, or any of its accepted forms, appears in an already normalised transcript."""
    return any(norm(w) and norm(w) in hyp_norm for w in (word if isinstance(word, list) else [word]))


def script_ok(text: str, lang: str | None) -> bool:
    """Every letter in the declared language's script: a Hebrew line holding one Latin letter fails."""
    want = SCRIPT.get(lang or '')
    if not want:
        return False
    return all(want in unicodedata.name(ch, '') for ch in text if unicodedata.category(ch).startswith('L'))
