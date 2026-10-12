"""The second half of a transcriber-plus-text-model pipeline: a text model turns a transcript into the same call.

Any OpenAI-compatible /chat/completions endpoint: OpenAI by default, or another (Groq, a local server) with
--base-url and --key-env. --model is required.

The request is, by default, a mechanical translation of request-template.json, so nothing of the prompt is
copied or edited. The system instruction becomes the system message. Each declaration becomes a tool, its
JSON Schema unchanged except one Gemini-only field, format: 'enum', which is dropped. Mode ANY becomes
tool_choice "required". Temperature and the output cap carry over. Only the audio part changes, into one
sentence: "The user said (transcribed; the transcriber heard language xx): «…»". A reference transcript has
no transcriber, so its sentence names no language. The text part after the audio stays.

--builder EXPORT asks your server's own function for the request instead (harness.Bridge): it is called as
EXPORT(context, transcript, heard, model) and returns the Chat Completions body your server would send.
That measures the text step exactly as built, rather than as translated.

--reasoning-effort sends reasoning_effort, for a model that reasons by default and bills it as output;
leave it out for a model that refuses it. A rate limit waits its retry-after up to a minute; a longer one,
or one for the day, stops the run (run.py keeps what ran).

--source reference (the default) reads the manifest's own transcripts, which measures the text model alone
and needs no recording. --source <transcriber> reads that transcriber's results (with the same --hints and
--vocab as its run), which completes the pipeline. Results go to results/llm_text--<model>--<source>.jsonl.
"""
from __future__ import annotations

import json
import os
import time

import requests

from harness import AdapterError, Clip, DailyLimit, Stopwatch, bridge, load_context, load_template, read_results, require, results_path, variant

NAME = 'llm_text'
KEYS = ()  # by --key-env, checked in prepare()
DEFAULT_BASE = 'https://api.openai.com/v1'

_request = None
_source: dict[tuple[str, int], dict] = {}


def key_env(opts) -> str:
    return getattr(opts, 'key_env', None) or 'OPENAI_API_KEY'


def model_slug(model: str) -> str:
    """A model id as part of a file name: provider/model becomes provider_model."""
    return model.replace('/', '_')


def output_name(opts) -> str:
    return f'{NAME}--{model_slug(opts.model)}--{opts.source}' + ('' if opts.source == 'reference' else variant(opts))


def needs_audio(opts) -> bool:
    return opts.source != 'reference'


def without_enum_format(schema):
    """The declarations' JSON Schema minus Gemini's format: 'enum', which Chat Completions does not take."""
    if isinstance(schema, dict):
        return {k: without_enum_format(v) for k, v in schema.items() if not (k == 'format' and v == 'enum')}
    if isinstance(schema, list):
        return [without_enum_format(v) for v in schema]
    return schema


def translated() -> tuple[str, list[dict], str, dict]:
    """The template's instruction, tools, closing text and generation settings, once."""
    global _request
    if _request is None:
        body = load_template()['body']
        parts = body['contents'][0]['parts']
        if len(parts) != 2 or 'inlineData' not in parts[0] or 'text' not in parts[1]:
            raise SystemExit("llm_text: the template's user turn is not [audio, text]; adapt translated() to your request")
        if body.get('toolConfig', {}).get('functionCallingConfig', {}).get('mode') != 'ANY':
            raise SystemExit('llm_text: the template does not force a call; adapt translated() to your request')
        tools = [
            {'type': 'function', 'function': {'name': d['name'], 'description': d['description'], 'parameters': without_enum_format(d['parameters'])}}
            for d in body['tools'][0]['functionDeclarations']
        ]
        _request = (body['systemInstruction']['parts'][0]['text'], tools, parts[1]['text'], body.get('generationConfig', {}))
    return _request


def prepare(opts) -> None:
    """Before the results file is opened: a model, a key, the source's results, and a template that translates."""
    if not opts.model:
        raise SystemExit('llm_text: name the model with --model')
    require(key_env(opts))
    load_source(opts)
    if getattr(opts, 'builder', None):
        if opts.builder not in bridge().exports:
            raise SystemExit(f'llm_text: the builder module has no export {opts.builder}')
    else:
        translated()


def load_source(opts) -> None:
    """The transcriber run this text step reads, with the same priming as this run's flags; the reference needs none."""
    if opts.source != 'reference':
        path = results_path(opts.source + variant(opts), opts.audio)
        if not path.exists():
            raise SystemExit(f'--source {opts.source}: {path.name} does not exist yet; run that transcriber first')
        for line in read_results(path):
            _source.setdefault((line['clip_id'], line['run']), line)


def heard(clip: Clip, opts) -> tuple[str, str | None, dict | None]:
    """The transcript to hand the model, the transcriber's language, and its results line (None for the reference)."""
    if opts.source == 'reference':
        return clip.transcript, None, None
    line = _source.get((clip.id, opts.run)) or _source.get((clip.id, 1))
    if not line or line.get('error') or line.get('transcript') is None:
        raise AdapterError(f'llm_text: no transcript from {opts.source} for {clip.id}' + (f' ({line["error"][:80]})' if line and line.get('error') else ''))
    return line['transcript'], line.get('language_detected'), line


def said(transcript: str, lang: str | None) -> str:
    return (f'The user said (transcribed; the transcriber heard language {lang}): «{transcript}»' if lang
            else f'The user said (transcribed): «{transcript}»')


def chat(body: dict, opts, watch: Stopwatch) -> dict:
    """One Chat Completions request. A rate limit waits its retry-after; one past a minute, or for the day, stops the run."""
    url = f'{(getattr(opts, "base_url", None) or DEFAULT_BASE).rstrip("/")}/chat/completions'
    for attempt in range(10):
        start = time.perf_counter()
        r = requests.post(url, headers={'Authorization': f'Bearer {os.environ[key_env(opts)]}'}, json=body, timeout=60)
        if r.status_code == 429:
            wait = float(r.headers.get('retry-after') or 2 ** min(attempt, 5))
            if wait > 60 or 'per day' in r.text.lower():
                raise DailyLimit(f'llm_text: rate-limited for {wait:.0f} s: {r.text[:160]}')
            time.sleep(wait)
            continue
        r.raise_for_status()
        watch.laps['request'] = round((time.perf_counter() - start) * 1000)
        return r.json()
    raise AdapterError('llm_text: still rate-limited after 10 tries')


def tool_calls(answer: dict) -> tuple[dict, list[dict]]:
    """The first choice and its tool calls as name and arguments; arguments that don't parse are kept as text."""
    choice = (answer.get('choices') or [{}])[0]
    calls = []
    for tc in (choice.get('message') or {}).get('tool_calls') or []:
        fn = tc.get('function') or {}
        try:
            args = json.loads(fn.get('arguments') or '{}')
        except json.JSONDecodeError:
            args = {'_unparsed': fn.get('arguments')}
        calls.append({'name': fn.get('name'), 'args': args})
    return choice, calls


def request_body(transcript: str, lang: str | None, opts) -> dict:
    if getattr(opts, 'builder', None):
        body = bridge().call(opts.builder, load_context(), transcript, lang, opts.model)
        return {**body, 'model': opts.model}
    instruction, tools, closing, gen = translated()
    body = {
        'model': opts.model,
        'messages': [
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': [{'type': 'text', 'text': said(transcript, lang)}, {'type': 'text', 'text': closing}]},
        ],
        'tools': tools,
        'tool_choice': 'required',
    }
    if 'temperature' in gen:
        body['temperature'] = gen['temperature']
    if 'maxOutputTokens' in gen:
        body['max_completion_tokens'] = gen['maxOutputTokens']
    if getattr(opts, 'reasoning_effort', None):
        body['reasoning_effort'] = opts.reasoning_effort
    return body


def run(clip: Clip, opts) -> dict:
    transcript, lang, source_line = heard(clip, opts)
    body = request_body(transcript, lang, opts)
    watch = Stopwatch()
    answer = chat(body, opts, watch)
    choice, calls = tool_calls(answer)
    if not calls:
        raise AdapterError(f'llm_text: no tool call (finish_reason {choice.get("finish_reason")})')
    source_ms = source_line.get('latency_ms') if source_line else None
    text_ms = watch.laps.get('request', watch.total())
    return {
        'model': opts.model,
        'source': opts.source,
        'source_transcript': transcript,
        'source_language': lang,
        'language_detected': calls[0]['args'].get('language'),
        'transcript': None,
        'function_call': calls[0],
        'function_calls': calls,
        'finish_reason': choice.get('finish_reason'),
        'latency_ms': text_ms,
        'source_latency_ms': source_ms,
        'pipeline_latency_ms': text_ms + source_ms if source_ms is not None else None,
        'timing_ms': watch.laps,
        'usage': answer.get('usage'),
        'raw': answer,
    }
