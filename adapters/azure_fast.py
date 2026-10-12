"""Azure AI Speech fast transcription: one synchronous multipart request.

The request is Microsoft's own (fast-transcription-create, api-version 2025-10-15): the audio, and a
definition naming the candidate locales. Azure's multilingual model, used when no locale is named, does not
cover every language (read its list for yours), so the adapter names candidates and Azure identifies the
language among them: by default every language in the context's "languages", or --hints to narrow them.
Azure always answers with one of the candidates, even for a language that is not among them. Profanity is
left unmasked so the transcript is what was said. --vocab sends a phrase list.

Fast transcription does not keep the audio, by Microsoft's documentation; check it for your region.

    python run.py --provider azure_fast --repeat 2
"""
from __future__ import annotations

import json
import os

import requests

from harness import AdapterError, Clip, Stopwatch, load_context, variant, vocabulary

NAME = 'azure_fast'
MODEL = 'fast-transcription-2025-10-15'
KEYS = ('AZURE_SPEECH_KEY', 'AZURE_SPEECH_ENDPOINT')
API_VERSION = '2025-10-15'

# One fast-transcription locale per language; check each on Azure's language-support page.
LOCALE = {'en': 'en-US', 'he': 'he-IL', 'ar': 'ar-SA', 'es': 'es-ES', 'fr': 'fr-FR', 'pt': 'pt-BR', 'ru': 'ru-RU',
          'uk': 'uk-UA', 'zh': 'zh-CN', 'de': 'de-DE', 'it': 'it-IT', 'ja': 'ja-JP'}


def output_name(opts) -> str:
    return NAME + variant(opts)


def run(clip: Clip, opts) -> dict:
    codes = opts.hints or load_context().get('languages') or ['en']
    definition = {'locales': [LOCALE[c] for c in codes], 'profanityFilterMode': 'None'}
    terms = vocabulary(getattr(opts, 'vocab', None))
    if terms:
        definition['phraseList'] = {'phrases': terms}
    endpoint = os.environ['AZURE_SPEECH_ENDPOINT'].rstrip('/')
    watch = Stopwatch()
    with clip.path.open('rb') as f:
        r = requests.post(
            f'{endpoint}/speechtotext/transcriptions:transcribe',
            params={'api-version': API_VERSION},
            headers={'Ocp-Apim-Subscription-Key': os.environ['AZURE_SPEECH_KEY']},
            data={'definition': json.dumps(definition)},
            files={'audio': (clip.path.name, f, 'audio/mp4')},
            timeout=60,
        )
    r.raise_for_status()
    body = r.json()
    watch.lap('request')
    if 'combinedPhrases' not in body:
        raise AdapterError(f'azure_fast: no combinedPhrases in the answer (keys: {", ".join(body)})')

    # Each language's share of the letters, from the per-phrase locales.
    counts: dict[str, int] = {}
    for p in body.get('phrases', []):
        code = (p.get('locale') or '')[:2].lower()
        if code:
            counts[code] = counts.get(code, 0) + sum(ch.isalpha() for ch in p.get('text', ''))
    total = sum(counts.values()) or 1
    langs = {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
    return {
        'model': MODEL,
        'candidates': definition['locales'],
        'language_detected': next(iter(langs), None),
        'languages': langs,
        'transcript': ' '.join(p.get('text', '') for p in body['combinedPhrases']).strip(),
        'function_call': None,
        'latency_ms': watch.total(),
        'timing_ms': watch.laps,
        'usage': {'audio_duration_ms': body.get('durationMilliseconds')},
        'raw': body,
    }
