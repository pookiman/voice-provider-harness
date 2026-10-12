"""What run.py, tts.py and score.py share: keys, the manifest, the context, the audio sets, the request template,
clip durations, results files, and the bridge to your own request builder."""
from __future__ import annotations

import atexit
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / 'examples'
CLIPS = HERE / 'clips'
TTS_CLIPS = CLIPS / 'tts'            # spoken by tts.py, one file per manifest line: clips/tts/<clip_id>.m4a
REAL_CLIPS = CLIPS / 'real'          # recordings, one folder per speaker: clips/real/<speaker>/<file>
SPEAKER = re.compile(r'^[a-z0-9]+$')
AUDIO_SETS = ('tts', 'real')
RESULTS = HERE / 'results'
TEMPLATE_FILE = HERE / 'request-template.json'
LEXICON_FILE = HERE / 'lexicon.txt'


def _own_or_example(name: str, example: str) -> Path:
    """Your own file at the top of the harness when there is one, else the shipped example."""
    own = HERE / name
    return own if own.exists() else EXAMPLES / example


MANIFEST_FILE = _own_or_example('manifest.csv', 'manifest.example.csv')
CONTEXT_FILE = _own_or_example('context.json', 'context.example.json')

# Windows consoles default to a code page that cannot print Hebrew or Arabic.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')


class AdapterError(Exception):
    """A provider answered, but not with a usable result. The message is kept, so it must hold no key and no audio."""


class DailyLimit(AdapterError):
    """A provider refused for a limit that will not clear within a minute. run.py stops at once and keeps what ran."""


def require(*names: str) -> None:
    """Keys come from this shell's environment, never from a file. Only the names are ever reported."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f'Set {", ".join(missing)} in this shell first (README, Keys).')


TAG = re.compile(r'^[a-z0-9-]+$')


@dataclass
class Clip:
    id: str
    path: Path
    language: str
    synthetic: bool
    transcript: str
    expected_function: str
    slots: dict
    notes: str
    speaker: str
    tts_text: str = ''
    tags: set[str] = field(default_factory=set)
    duration: float | None = None

    @property
    def exists(self) -> bool:
        return self.path.exists()


def audio_set(audio: str, speaker: str | None) -> str:
    """The set a run or a scoring reads: 'tts', or 'real-<speaker>' for one speaker's recordings."""
    if audio == 'tts':
        if speaker:
            raise SystemExit('--speaker goes with --audio real')
        return 'tts'
    if not speaker:
        have = real_speakers()
        raise SystemExit('--audio real needs --speaker: ' + (', '.join(have) if have else f'no folder in {REAL_CLIPS} yet'))
    if not SPEAKER.match(speaker):
        raise SystemExit(f'--speaker {speaker!r}: lower-case letters and digits only, the folder name under clips/real/')
    return f'real-{speaker}'


def real_speakers() -> list[str]:
    """The speakers with a folder of recordings."""
    return sorted(p.name for p in REAL_CLIPS.iterdir() if p.is_dir() and SPEAKER.match(p.name)) if REAL_CLIPS.exists() else []


def load_manifest(audio: str = 'tts') -> list[Clip]:
    """The manifest (README, The manifest), its slots parsed, each clip's path in the chosen audio set. A note's
    one-word parts before the prose, separated by semicolons, are its tags (codeswitch, edge, other, ambiguous…)."""
    clips = []
    with MANIFEST_FILE.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            parts = [p.strip() for p in (r.get('notes') or '').split(';')]
            clips.append(Clip(
                id=r['clip_id'],
                path=TTS_CLIPS / f'{r["clip_id"]}.m4a' if audio == 'tts' else REAL_CLIPS / audio.removeprefix('real-') / Path(r['file']).name,
                language=r['language'],
                synthetic=r.get('synthetic') == '1',
                transcript=r['transcript'],
                expected_function=r['expected_function'],
                slots=json.loads(r['expected_slots'] or '{}'),
                notes=r.get('notes') or '',
                speaker=r.get('speaker') or '',
                tts_text=r.get('tts_text') or '',
                tags={p for p in parts if TAG.match(p)},
            ))
    return clips


def load_context() -> dict:
    return json.loads(CONTEXT_FILE.read_text(encoding='utf-8'))


def compact(value) -> str:
    """JSON.stringify's bytes: no spaces, non-ASCII as itself."""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def load_template() -> dict:
    """request-template.json, after proving Python serialises its body to the bytes dump-request.mjs hashed."""
    if not TEMPLATE_FILE.exists():
        raise SystemExit('request-template.json is missing: run  node dump-request.mjs  (README, Plugging in your own contract)')
    t = json.loads(TEMPLATE_FILE.read_text(encoding='utf-8'))
    if hashlib.sha256(compact(t['body']).encode('utf-8')).hexdigest() != t['bodySha256']:
        raise SystemExit('request-template.json: the body does not serialise to the bytes that were hashed; regenerate it, never edit it')
    return t


_probed: dict[Path, dict] = {}


def probe(path: Path) -> dict:
    """Duration in seconds and the audio codec, by ffprobe."""
    if path not in _probed:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_name,sample_rate,channels', '-of', 'json', str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        j = json.loads(out)
        stream = (j.get('streams') or [{}])[0]
        _probed[path] = {
            'seconds': float(j.get('format', {}).get('duration') or 0),
            'codec': stream.get('codec_name'),
            'sample_rate': stream.get('sample_rate'),
            'channels': stream.get('channels'),
        }
    return _probed[path]


class Stopwatch:
    """Wall-clock laps in milliseconds, so an asynchronous provider's upload, create and poll show apart."""

    def __init__(self) -> None:
        self.start = self.last = time.perf_counter()
        self.laps: dict[str, int] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = self.laps.get(name, 0) + round((now - self.last) * 1000)
        self.last = now

    def total(self) -> int:
        return round((time.perf_counter() - self.start) * 1000)


# ─── Priming the transcribers ────────────────────────────────────────────────
#
# 'names': the people in the context as the data spells them, and their other-script spellings from the
# context's "aliases". 'full': those, plus one term a line from lexicon.txt, if you keep one. A lexicon that
# holds the very words in your test set is a ceiling, not a result: it can also put words in a transcript
# that were never said.

def vocabulary(kind: str | None) -> list[str]:
    if not kind:
        return []
    ctx = load_context()
    aliases = ctx.get('aliases') or {}
    names = [n for m in ctx.get('members', []) for n in (m, aliases.get(m)) if n]
    if kind == 'names':
        return names
    lexicon = [t.strip() for t in LEXICON_FILE.read_text(encoding='utf-8').splitlines() if t.strip()] if LEXICON_FILE.exists() else []
    return names + lexicon


def variant(opts) -> str:
    """The suffix a primed run's results carry (+hints, +own, +names, +full), so no primed run overwrites a plain one."""
    hints = getattr(opts, 'hints', None)
    return ('' if not hints else '+own' if hints == ['own'] else '+hints') + (f'+{opts.vocab}' if getattr(opts, 'vocab', None) else '')


def results_path(name: str, audio: str | None = None) -> Path:
    """results/<name>.jsonl for the TTS set, or with no audio; a speaker's recordings' results carry --real-<speaker>."""
    return RESULTS / f'{name}{"" if audio in (None, "tts") else f"--{audio}"}.jsonl'


def read_results(name_or_path) -> list[dict]:
    p = Path(name_or_path)
    if not p.suffix:
        p = results_path(str(name_or_path))
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def tts_voices() -> dict[str, dict]:
    """Who spoke each TTS clip, as tts.py logged it."""
    p = TTS_CLIPS / 'voices.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


# ─── The bridge: your server's own builders, called live ─────────────────────

class Bridge:
    """`dump-request.mjs <module> --serve`: one JSON line out, one back, so an adapter can ask your server's own
    function for a request instead of Python rebuilding it. HARNESS_MODULE names the module (default: the
    example builder); HARNESS_NODE the command that runs the script (default: node, which strips TypeScript
    types itself from Node 22.18; use "npx tsx" for older Node or for TypeScript that needs a compiler)."""

    def __init__(self, module: str | None = None):
        module = module or os.environ.get('HARNESS_MODULE') or str(EXAMPLES / 'request-builder.example.ts')
        cmd = shlex.split(os.environ.get('HARNESS_NODE', 'node')) + [str(HERE / 'dump-request.mjs'), module, '--serve']
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
        atexit.register(self.close)
        self.lock = threading.Lock()
        self.exports = self.ask({'op': 'exports'})['exports']

    def ask(self, req: dict) -> dict:
        with self.lock:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + '\n')
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise SystemExit('the bridge (dump-request.mjs --serve) stopped; its error is above')
        out = json.loads(line)
        if 'error' in out:
            raise AdapterError(f'bridge: {out["error"]}')
        return out

    def call(self, export: str, *args):
        """Your module's export, called with these JSON arguments; its result, awaited if it is a promise."""
        return self.ask({'op': 'call', 'export': export, 'args': list(args)})['result']

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


_bridge: Bridge | None = None


def bridge() -> Bridge:
    """One bridge per run, started on first use."""
    global _bridge
    if _bridge is None:
        _bridge = Bridge()
    return _bridge
