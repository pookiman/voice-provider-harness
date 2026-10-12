"""An audio-native model, Gemini's generateContent: the template's body, byte for byte, with the clip inline.

The body is request-template.json's, which dump-request.mjs took from your request builder, and
harness.load_template() has already proved Python serialises it to the bytes the builder made. The only
change is the clip's base64 where the placeholder was. The model is the template's, or --model's to try
another with the same body. The key travels in the x-goog-api-key header, never in the URL.

The calls are recorded as the model returned them, before anything your server would do to them afterwards;
score.py judges them as your app would apply them. Inline audio is not stored as a file, so there is nothing
to delete afterwards; what the provider keeps of a request is set by its terms, which a person reads.

    python run.py --provider gemini --repeat 2
"""
from __future__ import annotations

import base64
import json
import os

import requests

from harness import AdapterError, Clip, Stopwatch, compact, load_template

NAME = 'gemini'
KEYS = ('GEMINI_API_KEY',)
BASE = 'https://generativelanguage.googleapis.com'

_template = None


def output_name(opts) -> str:
    """gemini, or gemini--<model> for a model other than the template's."""
    model = getattr(opts, 'model', None)
    return NAME if not model or model == template()['model'] else f'{NAME}--{model}'


def template() -> dict:
    global _template
    if _template is None:
        _template = load_template()
    return _template


def body_for(clip: Clip) -> bytes:
    """The template's body with this clip's audio in it, as bytes."""
    t = template()
    text = compact(t['body'])
    slot = json.dumps(t['audioPlaceholder'])
    if text.count(slot) != 1:
        raise AdapterError('gemini: the template should hold the audio placeholder exactly once')
    audio = base64.b64encode(clip.path.read_bytes()).decode('ascii')
    return text.replace(slot, json.dumps(audio)).encode('utf-8')


def run(clip: Clip, opts) -> dict:
    model = getattr(opts, 'model', None) or template()['model']
    body = body_for(clip)
    watch = Stopwatch()
    r = requests.post(
        f'{BASE}/v1beta/models/{model}:generateContent',
        headers={'Content-Type': 'application/json', 'x-goog-api-key': os.environ['GEMINI_API_KEY']},
        data=body,
        timeout=60,
    )
    r.raise_for_status()
    answer = r.json()
    watch.lap('request')

    candidate = (answer.get('candidates') or [{}])[0]
    calls = [
        {'name': p['functionCall'].get('name'), 'args': p['functionCall'].get('args') or {}}
        for p in (candidate.get('content') or {}).get('parts') or []
        if isinstance(p, dict) and isinstance(p.get('functionCall'), dict)
    ]
    if not calls:
        raise AdapterError(f'gemini: no function call (finishReason {candidate.get("finishReason")})')
    return {
        'model': model,
        'language_detected': calls[0]['args'].get('language'),
        'transcript': None,
        'function_call': calls[0],
        'function_calls': calls,
        'finish_reason': candidate.get('finishReason'),
        'latency_ms': watch.total(),
        'timing_ms': watch.laps,
        'usage': answer.get('usageMetadata'),
        'raw': answer,
    }
