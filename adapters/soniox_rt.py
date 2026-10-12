"""Soniox's real-time transcription over a WebSocket, for a finished recording or one streamed as it is spoken.

The flow is Soniox's own: open wss://stt-rt.soniox.com/transcribe-websocket with the key in the
Authorization header; send the configuration as the first text frame, then the audio as binary frames, then
an empty text frame to end; read the final tokens until "finished".

Audio. A stream detects AAC in an ADTS stream but not in the m4a/mp4 container that phones record into, so
each clip is first repackaged as ADTS by ffmpeg: the same audio frames, no re-encoding, done before the clock
starts. A server would have to do the same (a few dozen lines walk the MP4 boxes, or ffmpeg -c:a copy -f adts).

Pace.
- --pace fast (the default) sends the whole clip at once, as an app that records and then sends would.
  latency_ms is the whole request, connection included. Even sent whole, a stream works through the audio
  at close to real time, so the time grows with the recording's length; set your server's timeout from it.
- --pace live sends it in 120 ms chunks at real-time speed, as an app streaming while the person speaks
  would; latency_ms is then the wait after the end of the audio, which is what the person waits through.
Results go to results/soniox_rt.jsonl, or soniox_rt_live.jsonl.

A stream creates no file and no transcription object, so there is nothing to delete afterwards.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from websockets.sync.client import connect

from adapters.soniox import DOMAIN, languages
from harness import AdapterError, Clip, Stopwatch, probe, variant, vocabulary

NAME = 'soniox_rt'
MODEL = 'stt-rt-v5'
KEYS = ('SONIOX_API_KEY',)
URL = 'wss://stt-rt.soniox.com/transcribe-websocket'
CHUNK_S = 0.12


def live(opts) -> bool:
    return getattr(opts, 'pace', 'fast') == 'live'


def output_name(opts) -> str:
    return ('soniox_rt_live' if live(opts) else 'soniox_rt') + variant(opts)


def adts(clip: Clip) -> bytes:
    """The clip's AAC frames as an ADTS stream, which a Soniox stream can detect; the m4a container it cannot."""
    return subprocess.run(['ffmpeg', '-v', 'error', '-i', str(clip.path), '-c:a', 'copy', '-f', 'adts', 'pipe:1'],
                          capture_output=True, check=True).stdout


def stream(config: dict, audio: bytes, seconds: float | None, paced: bool):
    """One stream: the configuration, the audio (whole, or paced), an empty frame for its end, then the final
    tokens until Soniox has finished. The clock starts before the connection."""
    finals: list[dict] = []
    last: dict = {}
    watch = Stopwatch()
    with connect(URL, additional_headers={'Authorization': f'Bearer {os.environ["SONIOX_API_KEY"]}'},
                 open_timeout=30, close_timeout=10, max_size=None, max_queue=None) as ws:
        watch.lap('connect')
        ws.send(json.dumps(config))
        if paced:
            step = max(1, int(len(audio) / seconds * CHUNK_S)) if seconds else len(audio)
            for i in range(0, len(audio), step):
                ws.send(audio[i:i + step])
                time.sleep(CHUNK_S)
        else:
            ws.send(audio)
        ws.send('')
        watch.lap('send')
        ended = time.perf_counter()
        while True:
            msg = json.loads(ws.recv(timeout=60))
            if msg.get('error_code') is not None:
                raise AdapterError(f'soniox_rt: {msg.get("error_code")}: {msg.get("error_message")}')
            finals += [t for t in msg.get('tokens', []) if t.get('is_final')]
            last = msg
            if msg.get('finished'):
                break
        after_end = round((time.perf_counter() - ended) * 1000)
        watch.lap('finalize')
    return finals, last, after_end, watch.total(), watch


def run(clip: Clip, opts) -> dict:
    audio = adts(clip)
    seconds = probe(clip.path)['seconds']
    config = {'model': MODEL, 'audio_format': 'auto', 'enable_language_identification': True}
    if opts.hints:
        config['language_hints'] = opts.hints
    terms = vocabulary(getattr(opts, 'vocab', None))
    if terms:
        config['context'] = {'general': [{'key': 'domain', 'value': DOMAIN}], 'terms': terms}
    finals, last, after_end, total, watch = stream(config, audio, seconds, live(opts))
    langs = languages(finals)
    return {
        'model': MODEL,
        'pace': 'live' if live(opts) else 'fast',
        'language_detected': next(iter(langs), None),
        'languages': langs,
        # A control token such as <end> carries no speech.
        'transcript': ''.join(t.get('text', '') for t in finals if not str(t.get('text', '')).startswith('<')).strip(),
        'function_call': None,
        'latency_ms': after_end if live(opts) else total,
        'after_end_ms': after_end,
        'stream_ms': total,
        'timing_ms': watch.laps,
        'usage': {'audio_duration_ms': last.get('final_audio_proc_ms')},
        'raw': {'final': last, 'tokens': finals},
    }
