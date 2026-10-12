"""The text step with a Claude model, through Anthropic's Messages API.

The same mechanical translation of request-template.json as llm_text, whose translation and transcript
loading this reuses. The system instruction becomes `system`, marked for prompt caching so cost compares
like with like against providers that cache by themselves; usage shows whether it was cached. Each
declaration becomes a tool with its JSON Schema as `input_schema`, minus Gemini's format: 'enum'. Mode ANY
becomes tool_choice {"type": "any"}, which forces a call; check that the model you name accepts it, as some
models with extended thinking do not. Temperature and the output cap carry over; no thinking is requested.

    python run.py --provider llm_claude --model <a Claude model id> [--source soniox]

Results go to results/llm_text--<model>--<source>.jsonl, beside the other text models, so score.py ranks
them together. Key: ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os

import requests

from adapters import llm_text
from harness import AdapterError, Clip, Stopwatch, require

NAME = 'llm_text'
KEYS = ('ANTHROPIC_API_KEY',)
URL = 'https://api.anthropic.com/v1/messages'


def output_name(opts) -> str:
    return llm_text.output_name(opts)


def needs_audio(opts) -> bool:
    return llm_text.needs_audio(opts)


def prepare(opts) -> None:
    if not opts.model:
        raise SystemExit('llm_claude: name the model with --model')
    require(*KEYS)
    llm_text.load_source(opts)
    llm_text.translated()


def run(clip: Clip, opts) -> dict:
    instruction, openai_tools, closing, gen = llm_text.translated()
    transcript, lang, source_line = llm_text.heard(clip, opts)
    body = {
        'model': opts.model,
        'max_tokens': gen.get('maxOutputTokens', 1024),
        'system': [{'type': 'text', 'text': instruction, 'cache_control': {'type': 'ephemeral'}}],
        'tools': [{'name': t['function']['name'], 'description': t['function']['description'], 'input_schema': t['function']['parameters']}
                  for t in openai_tools],
        'tool_choice': {'type': 'any'},
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': llm_text.said(transcript, lang)}, {'type': 'text', 'text': closing}]}],
    }
    if 'temperature' in gen:
        body['temperature'] = gen['temperature']
    watch = Stopwatch()
    r = requests.post(URL, headers={'x-api-key': os.environ['ANTHROPIC_API_KEY'], 'anthropic-version': '2023-06-01',
                                    'content-type': 'application/json'}, json=body, timeout=60)
    r.raise_for_status()
    answer = r.json()
    watch.lap('request')

    if answer.get('stop_reason') == 'refusal':
        raise AdapterError('llm_claude: the model refused the request')
    calls = [{'name': b.get('name'), 'args': b.get('input') or {}} for b in answer.get('content') or [] if b.get('type') == 'tool_use']
    if not calls:
        raise AdapterError(f'llm_claude: no tool call (stop_reason {answer.get("stop_reason")})')
    source_ms = source_line.get('latency_ms') if source_line else None
    return {
        'model': opts.model,
        'source': opts.source,
        'source_transcript': transcript,
        'source_language': lang,
        'language_detected': calls[0]['args'].get('language'),
        'transcript': None,
        'function_call': calls[0],
        'function_calls': calls,
        'finish_reason': answer.get('stop_reason'),
        'latency_ms': watch.total(),
        'source_latency_ms': source_ms,
        'pipeline_latency_ms': watch.total() + source_ms if source_ms is not None else None,
        'timing_ms': watch.laps,
        'usage': answer.get('usage'),
        'raw': answer,
    }
