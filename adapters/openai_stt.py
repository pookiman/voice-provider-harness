"""A transcriber behind an OpenAI-compatible /audio/transcriptions endpoint: one synchronous multipart request.

OpenAI's own by default; --base-url and --key-env point it at any compatible endpoint (Groq's Whisper is
one). No language is sent unless --hints is given, so the model's own detection is measured. The language
heard is read from either answer shape: OpenAI's list of detected languages, or a Whisper-style
verbose_json "language" name, which is mapped to its ISO code.

Vocabulary (--vocab names|full) goes where the endpoint takes it: OpenAI's keywords[] field by default, or a
Whisper-style prompt with --vocab-field prompt. A Whisper prompt is read as text said before the clip; one in
a single script can pull speech in another language into that script, so try both scripts, or none.

The file goes under its own name as audio/mp4. Endpoints differ in what they keep of an upload; read the
vendor's data-retention terms before you send anything, and send only invented data.

    python run.py --provider openai_stt --repeat 2
    python run.py --provider openai_stt --base-url https://api.groq.com/openai/v1 --key-env GROQ_API_KEY \\
        --model whisper-large-v3 --response-format verbose_json --vocab names --vocab-field prompt
"""
from __future__ import annotations

import os
import time

import requests

from harness import AdapterError, Clip, Stopwatch, require, variant, vocabulary

NAME = 'openai_stt'
KEYS = ()  # by --key-env, checked in prepare()
DEFAULT_MODEL = 'gpt-transcribe'
DEFAULT_BASE = 'https://api.openai.com/v1'
# A Whisper-style answer names the language in English words; the scorer reads ISO 639-1 codes.
LANGUAGES = {'hebrew': 'he', 'english': 'en', 'arabic': 'ar', 'russian': 'ru', 'ukrainian': 'uk', 'french': 'fr',
             'spanish': 'es', 'german': 'de', 'portuguese': 'pt', 'italian': 'it', 'chinese': 'zh', 'japanese': 'ja',
             'yiddish': 'yi', 'polish': 'pl', 'hungarian': 'hu', 'welsh': 'cy', 'greek': 'el', 'turkish': 'tr', 'dutch': 'nl'}


def key_env(opts) -> str:
    return getattr(opts, 'key_env', None) or 'OPENAI_API_KEY'


def model(opts) -> str:
    return getattr(opts, 'model', None) or DEFAULT_MODEL


def output_name(opts) -> str:
    """openai_stt, or openai_stt@<model> for another model, plus the priming suffix. The @ keeps the name one
    part when a text step's results name joins it with --."""
    m = model(opts)
    return (NAME if m == DEFAULT_MODEL else f'{NAME}@{m.replace("/", "_")}') + variant(opts)


def prepare(opts) -> None:
    require(key_env(opts))


def language_code(heard) -> str | None:
    if not isinstance(heard, str) or not heard.strip():
        return None
    h = heard.strip().lower()
    return h if len(h) <= 3 else LANGUAGES.get(h, h)


def transcribe(clip: Clip, fields: list[tuple[str, str]], watch: Stopwatch, url: str, key: str) -> dict:
    """One transcription: the form's fields, then the clip. A rate limit waits its retry-after, up to a minute."""
    for attempt in range(8):
        with clip.path.open('rb') as f:
            r = requests.post(url, headers={'Authorization': f'Bearer {os.environ[key]}'}, data=fields,
                              files={'file': (clip.path.name, f, 'audio/mp4')}, timeout=60)
        if r.status_code == 429:
            time.sleep(min(float(r.headers.get('retry-after') or 2 ** attempt), 60))
            continue
        r.raise_for_status()
        watch.lap('request')
        return r.json()
    raise AdapterError('openai_stt: still rate-limited after 8 tries')


def run(clip: Clip, opts) -> dict:
    fields: list[tuple[str, str]] = [('model', model(opts)), ('response_format', getattr(opts, 'response_format', None) or 'json')]
    for code in opts.hints or []:
        fields.append(('languages[]', code))
    terms = vocabulary(getattr(opts, 'vocab', None))
    if terms:
        if (getattr(opts, 'vocab_field', None) or 'keywords') == 'prompt':
            fields.append(('prompt', ', '.join(terms)))
        else:
            fields += [('keywords[]', t) for t in terms]
    url = f'{(getattr(opts, "base_url", None) or DEFAULT_BASE).rstrip("/")}/audio/transcriptions'
    watch = Stopwatch()
    body = transcribe(clip, fields, watch, url, key_env(opts))
    if 'text' not in body:
        raise AdapterError(f'openai_stt: no text in the answer (keys: {", ".join(body)})')
    codes = [lang.get('code') for lang in body.get('languages') or [] if isinstance(lang, dict) and lang.get('code')]
    heard = codes[0] if codes else language_code(body.get('language'))
    return {
        'model': model(opts),
        'language_detected': heard,
        'languages': codes,
        'transcript': body['text'].strip(),
        'function_call': None,
        'latency_ms': watch.total(),
        'timing_ms': watch.laps,
        'usage': body.get('usage') or ({'audio_duration_ms': round(body['duration'] * 1000)} if isinstance(body.get('duration'), (int, float)) else None),
        'raw': {k: v for k, v in body.items() if k != 'segments'},
    }
