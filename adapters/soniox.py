"""Soniox's asynchronous transcription: upload, create, poll, fetch, then delete both copies.

The flow is Soniox's own REST quickstart, with two changes. The poll waits a quarter of a second rather
than a full one, so the poll's granularity does not set the latency of a five-second clip. And the deletes
run after the clock stops, as a server would run them in the background.

An asynchronous job costs a flat two to three seconds whatever the clip's length, which only a clock
around the whole pipeline shows; the laps (upload, create, poll, fetch) say which step it is.

The uploaded file and the transcription are deleted after every clip, and a run's result line says
whether both deletes succeeded. run.py --leftovers soniox lists whatever the account still holds, and
--delete removes the copies named after a clip in the manifest. Language identification is on; --hints
sends language_hints. --vocab sends Soniox's context object: a domain line, and the terms to expect.

    python run.py --provider soniox --repeat 2
    python run.py --leftovers soniox --delete
"""
from __future__ import annotations

import os
import time

import requests

from harness import AdapterError, Clip, Stopwatch, load_manifest, variant, vocabulary

NAME = 'soniox'
MODEL = 'stt-async-v5'
KEYS = ('SONIOX_API_KEY',)
BASE = 'https://api.soniox.com'
DOMAIN = 'short voice commands to an app'
POLL_S = 0.25
GIVE_UP_MS = 120_000


def output_name(opts) -> str:
    return NAME + variant(opts)


def languages(tokens: list[dict]) -> dict[str, float]:
    """Each language's share of the transcript's letters, from the per-token tags."""
    counts: dict[str, int] = {}
    for t in tokens:
        lang = t.get('language')
        letters = sum(ch.isalpha() for ch in t.get('text', ''))
        if lang and letters:
            counts[lang] = counts.get(lang, 0) + letters
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def delete(s: requests.Session, url: str) -> bool:
    """A cleanup that never hides the error that sent us here."""
    try:
        return s.delete(url, timeout=30).ok
    except requests.RequestException:
        return False


def session() -> requests.Session:
    s = requests.Session()
    s.headers['Authorization'] = f'Bearer {os.environ["SONIOX_API_KEY"]}'
    return s


def leftovers(delete_ours: bool) -> list[str]:
    """What the account still holds, transcriptions then files; those named after a clip in the manifest are deleted on request."""
    s = session()
    ours = {c.path.name for c in load_manifest()}
    lines = []
    for kind in ('transcriptions', 'files'):
        cursor = None
        while True:
            r = s.get(f'{BASE}/v1/{kind}', params={'limit': 1000, **({'cursor': cursor} if cursor else {})}, timeout=30)
            r.raise_for_status()
            page = r.json()
            for item in page.get(kind, []):
                mine = item.get('filename') in ours
                gone = delete_ours and mine and delete(s, f'{BASE}/v1/{kind}/{item["id"]}')
                lines.append(f'{kind[:-1]:<13} {item["id"]}  {item.get("filename")}  {item.get("created_at")}'
                             f'  {"a clip of ours" if mine else "not ours, left alone"}{"  DELETED" if gone else ""}')
            cursor = page.get('next_page_cursor')
            if not cursor:
                break
    return lines


def run(clip: Clip, opts) -> dict:
    deleted: dict[str, bool] = {}
    try:
        return transcribe(clip, opts, deleted)
    except Exception as e:
        # The deletes have run by now; the error line still says whether the provider kept a copy.
        e.deleted = deleted
        raise


def transcribe(clip: Clip, opts, deleted: dict[str, bool]) -> dict:
    s = session()
    watch = Stopwatch()
    with clip.path.open('rb') as f:
        r = s.post(f'{BASE}/v1/files', files={'file': (clip.path.name, f, 'audio/mp4')}, timeout=60)
    r.raise_for_status()
    file_id = r.json()['id']
    watch.lap('upload')
    try:
        body = {'model': MODEL, 'file_id': file_id, 'enable_language_identification': True}
        if opts.hints:
            body['language_hints'] = opts.hints
        terms = vocabulary(getattr(opts, 'vocab', None))
        if terms:
            body['context'] = {'general': [{'key': 'domain', 'value': DOMAIN}], 'terms': terms}
        r = s.post(f'{BASE}/v1/transcriptions', json=body, timeout=30)
        r.raise_for_status()
        tid = r.json()['id']
        watch.lap('create')
        try:
            polls = 0
            while True:
                r = s.get(f'{BASE}/v1/transcriptions/{tid}', timeout=30)
                r.raise_for_status()
                status = r.json()
                polls += 1
                if status.get('status') == 'completed':
                    break
                if status.get('status') == 'error':
                    raise AdapterError(f'soniox: {status.get("error_type")}: {status.get("error_message")}')
                if watch.total() > GIVE_UP_MS:
                    raise AdapterError(f'soniox: still {status.get("status")} after {GIVE_UP_MS} ms')
                time.sleep(POLL_S)
            watch.lap('poll')
            r = s.get(f'{BASE}/v1/transcriptions/{tid}/transcript', timeout=30)
            r.raise_for_status()
            transcript = r.json()
            watch.lap('fetch')
            latency = watch.total()
        finally:
            deleted['transcription'] = delete(s, f'{BASE}/v1/transcriptions/{tid}')
    finally:
        deleted['file'] = delete(s, f'{BASE}/v1/files/{file_id}')

    langs = languages(transcript.get('tokens', []))
    return {
        'model': MODEL,
        'language_detected': next(iter(langs), None),
        'languages': langs,
        'transcript': transcript.get('text', ''),
        'function_call': None,
        'latency_ms': latency,
        'timing_ms': watch.laps,
        'polls': polls,
        'usage': {'audio_duration_ms': status.get('audio_duration_ms')},
        'deleted': deleted,
        'raw': {'status': status, 'transcript': transcript},
    }
