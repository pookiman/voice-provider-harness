"""Scores every results file against the manifest and drafts the scorecard: gates first, then a ranking.

    python score.py                                  the TTS set's results
    python score.py --audio real --speaker anna      one speaker's recordings
    python score.py --audio real                     every speaker, then a table by speaker
    python score.py --control gemini--<model>        another run as the control (default: gemini)

Writes results/scorecard[--real-<speaker>].md (the tables), results/human-pass[--real-<speaker>].md (every
spoken line beside what was said, for a listener to score 0-2) and results/scores[--real-<speaker>].json
(every clip's verdict, so any number can be traced to its clips).

Calls are judged the way an app would apply them: an assignee or attendee counts when it matches a member's
name, ignoring case, and nobody named means the speaker; a task without a date is due today; an event without
a duration lasts 60 minutes; a category of "other" counts as none given; a completion finishes the first
pending task whose title contains title_contains. Titles and names match at a normalised similarity of 0.8 or
more, or when the title holds every expected word. A provider error counts as a lost clip in every rate.

The context's "languages" name the hard language first (the one the slots and recall columns report on) and
the other main language second. Function names are the four constants below: change them to your contract's.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import jiwer

from harness import AUDIO_SETS, HERE, RESULTS, audio_set, load_context, load_manifest, read_results, real_speakers, results_path
from normalise import best_ratio, norm, script_ok, word_present

# The contract's four functions.
ADD, COMPLETE, SUMMARY, DECLINE = 'add_entries', 'complete_entry', 'speak_summary', 'decline'
MAX_SPOKEN_CHARS = 280
TRANSCRIBERS = ('openai_stt', 'soniox', 'soniox_rt', 'soniox_rt_live', 'azure_fast')
AUDIO_NATIVE = ('gemini',)

# List prices in US$, as read from the vendors' pages in October 2026. Prices change: read the page before you
# trust a cost. None means unread, and the cost column says so.
PRICE_STT = {  # by transcriber, or by model for openai_stt
    'soniox': {'per_audio_hour': 0.10},
    'soniox_rt': {'per_audio_hour': 0.12},
    'soniox_rt_live': {'per_audio_hour': 0.12},
    'azure_fast': {'per_audio_hour': None},
    'gpt-transcribe': {'per_audio_minute': 0.0045},
    'whisper-large-v3': {'per_audio_hour': 0.111, 'min_ms': 10_000},
    'whisper-large-v3-turbo': {'per_audio_hour': 0.04, 'min_ms': 10_000},
}
PRICE_AUDIO_NATIVE = {  # per 1M tokens
    'gemini-2.5-flash': {'text_in': 0.30, 'audio_in': 1.00, 'out': 2.50},
}
PRICE_TEXT = {  # per 1M tokens
    'claude-haiku-4-5': {'in': 1.00, 'cached_in': 0.10, 'cache_write': 1.25, 'out': 5.00},
    'openai/gpt-oss-120b': {'in': 0.15, 'cached_in': 0.075, 'out': 0.60},
    'openai/gpt-oss-20b': {'in': 0.075, 'cached_in': 0.037, 'out': 0.30},
}


# ─── Which runs there are ────────────────────────────────────────────────────

def base_of(name: str) -> str:
    """A transcriber run's adapter: soniox+hints+names → soniox, openai_stt@whisper-large-v3 → openai_stt."""
    return name.split('+')[0].split('@')[0]


def discover(audio: str, control: str) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """The transcriber runs; the candidates (the control first, then other audio-native runs, then every text model
    on every transcriber run); and the text models' runs on the reference transcripts."""
    stems = [p.name[:-len('.jsonl')] for p in RESULTS.glob('*.jsonl')]
    real = re.compile(r'--real(-[a-z0-9]+)?$')
    mine = [s[:-len(f'--{audio}')] for s in stems if s.endswith(f'--{audio}')] if audio != 'tts' else [s for s in stems if not real.search(s)]
    trans = sorted(n for n in mine if base_of(n) in TRANSCRIBERS and '--' not in n)
    native = sorted(n for n in mine if n.split('--')[0] in AUDIO_NATIVE and n != control)
    pipes = sorted((n.split('--')[2], n.split('--')[1], n) for n in mine
                   if n.startswith('llm_text--') and len(n.split('--')) == 3 and base_of(n.split('--')[2]) in TRANSCRIBERS)
    refs = sorted((s, s.split('--')[1]) for s in stems if s.startswith('llm_text--') and s.endswith('--reference'))
    candidates = ([(control, f'{control} (control)')] if control in mine else []) + [(n, n) for n in native]
    candidates += [(n, f'{t} → {m}') for t, m, n in pipes]
    return trans, candidates, refs


# ─── Calls, judged as an app applies them ────────────────────────────────────

class Judge:
    def __init__(self, ctx: dict):
        self.members = ctx['members']
        self.speaker = ctx['speaker']
        self.today = ctx['today']
        self.tasks = [t['title'] for t in ctx.get('tasks', [])]

    def people(self, names) -> set[str]:
        found = {m for n in (names or []) for m in self.members if isinstance(n, str) and m.lower() == n.lower()}
        return found or {self.speaker}

    def completes(self, title_contains: str | None) -> str | None:
        needle = (title_contains or '').lower()
        return next((t for t in self.tasks if needle and needle in t.lower()), None)

    def entries(self, expected: dict, actual: dict) -> tuple[int, int, list[str], dict]:
        """Fields right, fields scored, the misses, and the categories, for one add call's items, tasks and events.
        A category is kept apart from the slots: how often one is given, and how often a given one is right."""
        ok = total = 0
        misses: list[str] = []
        cat = {'expected': 0, 'given': 0, 'right': 0}
        for kind, key in (('items', 'name'), ('tasks', 'title'), ('events', 'title')):
            exp = expected.get(kind) or []
            act = [a for a in (actual.get(kind) or []) if isinstance(a, dict) and str(a.get(key) or '').strip()]
            used: set[int] = set()
            for e in exp:
                ratios = [(best_ratio(e[key], str(a.get(key))), i) for i, a in enumerate(act) if i not in used]
                ratio, i = max(ratios) if ratios else (0.0, -1)
                fields = [f for f in e if f not in (key, 'category')]
                total += 1 + len(fields)
                if ratio < 0.8:
                    misses.append(f'{kind[:-1]} missing: {e[key]}' + (f' (nearest {act[i][key]!r}, {ratio:.2f})' if i >= 0 else ''))
                    continue
                used.add(i)
                ok += 1
                a = act[i]
                if 'category' in e:
                    cat['expected'] += 1
                    if a.get('category') and a.get('category') != 'other':
                        cat['given'] += 1
                        cat['right'] += self.field('category', e['category'], a)
                for f in fields:
                    good = self.field(f, e[f], a)
                    ok += good
                    if not good:
                        misses.append(f'{kind[:-1]} {e[key] if isinstance(e[key], str) else e[key][0]}: {f} expected {e[f]!r}, got {a.get(f)!r}')
            for i, a in enumerate(act):
                if i not in used:
                    total += 1
                    misses.append(f'{kind[:-1]} extra: {a.get(key)!r}')
        return ok, total, misses, cat

    def field(self, f: str, want, a: dict) -> bool:
        allowed = want if isinstance(want, list) and f not in ('assigned_to', 'attendees') else [want]
        if f == 'quantity':
            try:
                got = float(a.get('quantity') if a.get('quantity') is not None else 1) or 1
            except (TypeError, ValueError):
                got = 1
            return got == float(want)
        if f == 'category':
            return a.get('category') != 'other' and a.get('category') in allowed
        if f in ('assigned_to', 'attendees'):
            return self.people(a.get(f)) == self.people(want)
        if f == 'due_date':
            return (str(a.get('due_date') or '')[:10] or self.today) in allowed
        if f == 'date':
            return str(a.get('date') or '').strip() in allowed
        if f == 'start_time':
            got = str(a.get('start_time') or '').strip()
            return (got.zfill(5) if len(got) in (4, 5) and ':' in got else None) == want
        if f == 'duration_minutes':
            try:
                got = round(float(a.get('duration_minutes'))) if float(a.get('duration_minutes') or 0) > 0 else 60
            except (TypeError, ValueError):
                got = 60
            return got == want
        return (a.get(f) or None) in allowed

    def failed(self, clip, error: str) -> dict:
        """A provider error is a clip the person loses: it counts, as a failure, in every rate."""
        v = {'function': None, 'n_calls': 0, 'language': None, 'language_ok': False if clip.language else None,
             'function_ok': False, 'slots_ok': False, 'full': False, 'error': error, 'misses': [f'provider error: {error[:120]}']}
        if clip.expected_function == DECLINE:
            v['declined_ok'] = False
        return v

    def calls(self, clip, calls: list[dict]) -> dict:
        names = [c.get('name') for c in calls]
        first = calls[0] if calls else {'name': None, 'args': {}}
        args = first.get('args') or {}
        lang = args.get('language')
        v = {'function': first.get('name'), 'n_calls': len(calls), 'language': lang,
             'language_ok': (lang == clip.language) if clip.language else None,
             'function_ok': first.get('name') == clip.expected_function, 'misses': []}
        exp = clip.expected_function
        if exp == DECLINE:
            v['declined_ok'] = bool(calls) and all(n == DECLINE for n in names)
            v['slots_ok'] = v['declined_ok']
        else:
            v['declined_wrongly'] = first.get('name') == DECLINE
            if exp == ADD:
                adds = [c.get('args') or {} for c in calls if c.get('name') == ADD]
                merged = {k: [x for a in adds for x in (a.get(k) or [])] for k in ('items', 'tasks', 'events')}
                ok, total, misses, cat = self.entries(clip.slots, adds[0] if adds else {})
                ok_all, total_all, _, _ = self.entries(clip.slots, merged)
                v.update(fields_ok=ok, fields_total=total, misses=misses, categories=cat,
                         slots_ok=v['function_ok'] and ok == total, slots_ok_across_calls=ok_all == total_all)
            elif exp == COMPLETE:
                hit = self.completes(args.get('title_contains')) if first.get('name') == COMPLETE else None
                v.update(completes=hit, slots_ok=hit == clip.slots.get('task'))
                if hit != clip.slots.get('task'):
                    v['misses'] = [f'completes {hit!r}, expected {clip.slots.get("task")!r} (title_contains {args.get("title_contains")!r})']
            else:
                v['slots_ok'] = v['function_ok']
        line = args.get('confirmation') if first.get('name') in (ADD, COMPLETE) else args.get('message') if first.get('name') == SUMMARY else None
        if line is not None:
            v['line'] = line
            v['line_script_ok'] = script_ok(line, lang)
            v['line_length_ok'] = len(line) <= MAX_SPOKEN_CHARS
        v['full'] = bool((v['language_ok'] is not False) and v['function_ok'] and v.get('slots_ok', True))
        # A wrong write: the line is wrong AND the calls would still add or change something the person did not ask for.
        writes = any(c.get('name') == ADD and any((c.get('args') or {}).get(k) for k in ('items', 'tasks', 'events')) for c in calls)
        completes_other = first.get('name') == COMPLETE and v.get('completes') not in (None, clip.slots.get('task'))
        v['wrong_write'] = not v['full'] and (writes and exp != ADD or (exp == ADD and writes and not v.get('slots_ok')) or completes_other)
        return v


def verdict(judge: Judge, clip, r: dict) -> dict:
    if r.get('error'):
        return judge.failed(clip, r['error'])
    return judge.calls(clip, r.get('function_calls') or [])


# ─── Aggregates ──────────────────────────────────────────────────────────────

def pct(xs) -> str:
    xs = [x for x in xs if x is not None]
    return f'{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})' if xs else '–'


def rate(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def quant(values, q: float) -> str:
    v = sorted(x for x in values if x is not None)
    return f'{v[min(len(v) - 1, int(round(q * (len(v) - 1))))] / 1000:.1f} s' if v else '–'


def subset(clip, name: str, hard: str, other: str) -> bool:
    t = clip.tags
    return {
        'hard': clip.language == hard and 'codeswitch' not in t and 'edge' not in t,
        'second': clip.language == other and 'codeswitch' not in t,
        'cs': 'codeswitch' in t,
        'other': 'other' in t,
        'clean3': 'edge' not in t and clip.language != '' and (clip.duration or 0) >= 3,
        'unambiguous': 'ambiguous' not in t,
        'all': True,
    }[name]


def by_clip_run(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(r['clip_id'], r['run']): r for r in rows}


def flips(rows: list[dict], key) -> str:
    """The share of clips whose answer changed between run 1 and run 2: anything smaller is weather."""
    runs = by_clip_run(rows)
    pairs = [(runs[(c, 1)], runs[(c, 2)]) for (c, run) in runs if run == 1 and (c, 2) in runs]
    if not pairs:
        return 'one run'
    changed = sum(key(a) != key(b) for a, b in pairs)
    return f'{100 * changed / len(pairs):.0f}% ({changed}/{len(pairs)})'


def transcriber_cost(r: dict) -> float | None:
    """One clip's transcription charge, from the provider's own usage where it reports one."""
    p = PRICE_STT.get(r.get('model')) or PRICE_STT.get(r.get('provider'))
    if not p:
        return None
    u = r.get('usage') or {}
    if 'per_audio_minute' in p:
        secs = u.get('seconds') if u.get('type') == 'duration' else r.get('duration_s')
        return (secs or 0) / 60 * p['per_audio_minute']
    if p.get('per_audio_hour') is None:
        return None
    ms = max(u.get('audio_duration_ms') or (r.get('duration_s') or 0) * 1000, p.get('min_ms', 0))
    return ms / 3_600_000 * p['per_audio_hour']


def call_cost(r: dict) -> float | None:
    """One model call's charge, from its usage fields: Gemini's modalities, Anthropic's cache reads and writes,
    or Chat Completions' cached prompt tokens."""
    u = r.get('usage') or {}
    if 'promptTokensDetails' in u or 'candidatesTokenCount' in u:
        p = PRICE_AUDIO_NATIVE.get(r.get('model'))
        if not p:
            return None
        det = {d.get('modality'): d.get('tokenCount', 0) for d in u.get('promptTokensDetails') or []}
        return (det.get('TEXT', 0) * p['text_in'] + det.get('AUDIO', 0) * p['audio_in'] + u.get('candidatesTokenCount', 0) * p['out']) / 1e6
    p = PRICE_TEXT.get(r.get('model'))
    if p is None:
        return None
    if 'input_tokens' in u:
        return (u.get('input_tokens', 0) * p['in'] + (u.get('cache_read_input_tokens') or 0) * p['cached_in']
                + (u.get('cache_creation_input_tokens') or 0) * p.get('cache_write', p['in']) + u.get('output_tokens', 0) * p['out']) / 1e6
    cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    return ((u.get('prompt_tokens', 0) - cached) * p['in'] + cached * p['cached_in'] + u.get('completion_tokens', 0) * p['out']) / 1e6


def names_heard(ctx: dict, clips: dict, rows: list[dict], run: int) -> tuple[int, int, list[str]]:
    """Members named in a line, found in its transcript in either script (the context's "aliases")."""
    aliases = ctx.get('aliases') or {}
    forms = {m: [m] + ([aliases[m]] if aliases.get(m) else []) for m in ctx.get('members', [])}
    right = total = 0
    misses: list[str] = []
    for r in rows:
        if r['run'] != run or r.get('error'):
            continue
        said = norm(clips[r['clip_id']].transcript)
        hyp = norm(r.get('transcript'))
        for m, fs in forms.items():
            if any(norm(f) in said.split() for f in fs):
                total += 1
                if any(norm(f) in hyp.split() for f in fs):
                    right += 1
                else:
                    misses.append(f'{r["clip_id"]} "{(r.get("transcript") or "")[:50]}"')
    return right, total, misses


# ─── The listening pass ──────────────────────────────────────────────────────

def human_scores(path: Path) -> dict[tuple[str, str], str]:
    """The Score cells already filled in on a human-pass page, by clip and candidate, kept as written."""
    scores: dict[tuple[str, str], str] = {}
    if not path.exists():
        return scores
    cid = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('### '):
            cid = line[4:].split(' ')[0]
        elif cid and line.startswith('| ') and not line.startswith('| Candidate') and not line.startswith('|---'):
            cells = line.split('|')
            if len(cells) >= 6 and cells[4].strip():
                scores[(cid, cells[1].strip())] = cells[4].strip()
    return scores


def score_value(cell: str) -> int | None:
    head = cell.strip()[:1]
    return int(head) if head in ('0', '1', '2') else None


# ─── The scorecard ───────────────────────────────────────────────────────────

def score_set(audio: str, control: str, echo: bool = True) -> dict:
    """Scores one audio set, tts or real-<speaker>, writes its three files, and returns its verdicts."""
    suffix = '' if audio == 'tts' else f'--{audio}'
    title = 'TTS clips' if audio == 'tts' else f"{audio.removeprefix('real-')}'s recordings"
    hp_path = RESULTS / f'human-pass{suffix}.md'
    kept = human_scores(hp_path)  # read before anything is rewritten

    ctx = load_context()
    langs = ctx.get('languages') or ['en']
    hard, second = langs[0], (langs[1] if len(langs) > 1 else langs[0])
    sub = lambda c, s: subset(c, s, hard, second)
    clips = {c.id: c for c in load_manifest(audio)}
    judge = Judge(ctx)
    out: list[str] = [f'# Scorecard draft — {title}', '', 'Generated by score.py: gates first, any miss removing the candidate, then the ranking.', '']
    scores: dict = {}

    def load(name: str, with_audio: bool = True) -> list[dict]:
        p = results_path(name, audio if with_audio else None)
        if not p.exists():
            return []
        rows = [r for r in read_results(p) if r['clip_id'] in clips]
        for r in rows:
            if r.get('duration_s') and clips[r['clip_id']].duration is None:
                clips[r['clip_id']].duration = r['duration_s']
        return rows

    # Transcribers: language, WER, content-word recall, flip rate, latency.
    trans_names, candidates, refs = discover(audio, control)
    trans = {t: load(t) for t in trans_names}
    out += ['## Transcribers', '',
            f'| Transcriber | Clips | Language | Language, code-switched | WER {hard} | WER {second} | WER cs | Recall {hard} | Recall {second} | Recall cs | Flip rate | p50 / p95 |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for t, rows in trans.items():
        if not rows:
            continue
        ok = [{**r, 'transcript': r.get('transcript') or ''} for r in rows]
        run1 = [r for r in ok if r['run'] == 1]
        lid = lambda s: pct([r.get('language_detected') == clips[r['clip_id']].language for r in run1 if sub(clips[r['clip_id']], s) and clips[r['clip_id']].language])

        def wer(s):
            pairs = [(norm(clips[r['clip_id']].transcript), norm(r['transcript'])) for r in run1 if sub(clips[r['clip_id']], s) and 'edge' not in clips[r['clip_id']].tags]
            pairs = [(a, b) for a, b in pairs if a]
            return f'{100 * jiwer.wer([a for a, _ in pairs], [b or "∅" for _, b in pairs]):.1f}%' if pairs else '–'

        def recall(s):
            return pct([word_present(w, norm(r['transcript'])) for r in run1 if sub(clips[r['clip_id']], s) for w in clips[r['clip_id']].slots.get('_words', [])])

        out.append(f'| {t} | {len(run1)} | {lid("all")} | {lid("cs")} | {wer("hard")} | {wer("second")} | {wer("cs")} | {recall("hard")} | {recall("second")} | {recall("cs")} | '
                   f'{flips(ok, lambda r: (r.get("language_detected"), norm(r.get("transcript"))))} | {quant([r.get("latency_ms") for r in ok], .5)} / {quant([r.get("latency_ms") for r in ok], .95)} |')
        confusion: dict[str, int] = {}
        for r in run1:
            c = clips[r['clip_id']]
            if c.language and r.get('language_detected') != c.language:
                k = f'{c.language}→{r.get("language_detected")}'
                confusion[k] = confusion.get(k, 0) + 1
        scores.setdefault('transcribers', {})[t] = {'confusion': confusion, 'errors': sum(1 for r in rows if r.get('error'))}
    conf = {t: v['confusion'] for t, v in scores.get('transcribers', {}).items() if v['confusion']}
    out += ['', 'Language confusions (expected→detected, run 1): ' + ('; '.join(f'{t}: {", ".join(f"{k} ×{n}" for k, n in c.items())}' for t, c in conf.items()) or 'none'), '']
    if trans and ctx.get('members'):
        out += ["**The members' names in the transcripts** (either script counts):", '', '| Transcriber | Run 1 | Run 2 | Run 1 misses |', '|---|---|---|---|']
        for t, rows in trans.items():
            cells = []
            for run in (1, 2):
                right, total, _ = names_heard(ctx, clips, rows, run)
                cells.append(f'{right} of {total}' if total else '–')
            misses = names_heard(ctx, clips, rows, 1)[2]
            out.append(f'| {t} | {cells[0]} | {cells[1]} | {"; ".join(misses[:6]) or "none"} |')
        out.append('')

    # Candidates: the control, the audio-native runs and the pipelines; then the text models alone.
    judged: dict[str, list[tuple[dict, dict]]] = {}
    for name, _ in candidates + refs:
        rows = load(name, with_audio=not name.endswith('--reference'))
        judged[name] = [(r, verdict(judge, clips[r['clip_id']], r)) for r in rows]
        scores.setdefault('clips', {})[name] = {f'{r["clip_id"]}#{r["run"]}': v for r, v in judged[name]}
        errors = [r for r in rows if r.get('error')]
        if errors:
            scores.setdefault('errors', {})[name] = [f'{r["clip_id"]}: {r["error"][:120]}' for r in errors]
    ctl = judged.get(control) or []

    def full_rate(pairs, s):
        return rate([v['full'] for r, v in pairs if r['run'] == 1 and sub(clips[r['clip_id']], s)])

    out += ['## Gates', '',
            f'| Candidate | Clips | Language, clean ≥ 3 s (≥ 95%) | Language, code-switched (100%) | Function, unambiguous (≥ 95%) | Off-topic declined (all) | '
            f'Real commands declined (none) | Spoken-line script (100%) | Within 2 points of control, {hard} / cs | All correct (not a gate) | Wrong writes (not a gate) |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, label in candidates:
        pairs = [(r, v) for r, v in judged.get(name, []) if r['run'] == 1]
        if not pairs:
            continue
        share = lambda s, k: pct([v[k] for r, v in pairs if sub(clips[r['clip_id']], s) and v.get(k) is not None])
        off = [v['declined_ok'] for r, v in pairs if clips[r['clip_id']].expected_function == DECLINE and clips[r['clip_id']].language]
        # An ambiguous clip (two voices at once, say) may fairly be declined; a clear request may not.
        wrongly = sum(1 for r, v in pairs if v.get('declined_wrongly') and 'ambiguous' not in clips[r['clip_id']].tags)
        lines = [v['line_script_ok'] for r, v in pairs if 'line_script_ok' in v]
        bar = []
        for s in ('hard', 'cs'):
            mine, theirs = full_rate(pairs, s), full_rate(ctl, s)
            bar.append('–' if mine is None or theirs is None else 'control' if name == control else
                       f'{"pass" if mine >= theirs - 0.02 else "FAIL"} ({100 * mine:.0f} vs {100 * theirs:.0f})')
        out.append(f'| {label} | {len(pairs)} | {share("clean3", "language_ok")} | {share("cs", "language_ok")} | {share("unambiguous", "function_ok")} | {pct(off)} | '
                   f'{wrongly} | {pct(lines)} | {" / ".join(bar)} | {pct([v["full"] for _, v in pairs])} | {sum(1 for _, v in pairs if v.get("wrong_write"))} |')

    out += ['', '## Ranking, in order', '',
            f'| Candidate | Slots right, {hard} | Content-word recall, {hard} | Flip rate | p95 end to end | Cost per 1,000 | One call, every entry | Category given / right when given |',
            '|---|---|---|---|---|---|---|---|']
    for name, label in candidates:
        rows_all = judged.get(name, [])
        pairs = [(r, v) for r, v in rows_all if r['run'] == 1]
        if not pairs:
            continue
        t = name.split('--')[-1] if name.startswith('llm_text--') else None
        slots = pct([v['slots_ok'] for r, v in pairs if sub(clips[r['clip_id']], 'hard') and clips[r['clip_id']].expected_function in (ADD, COMPLETE)])
        if t and trans.get(t):
            recall = pct([word_present(w, norm(r.get('transcript'))) for r in trans[t] if r['run'] == 1 and sub(clips[r['clip_id']], 'hard')
                          for w in clips[r['clip_id']].slots.get('_words', [])])
        else:
            recall = 'n/a (no transcript)'
        lat = [r.get('pipeline_latency_ms') if t else r.get('latency_ms') for r, _ in rows_all]
        costs = []
        tr_by = by_clip_run(trans.get(t, [])) if t else {}
        for r, _ in pairs:
            if r.get('error'):
                continue  # a failed request bills nothing; its clip already counts as lost
            c = call_cost(r)
            if t:
                tr = tr_by.get((r['clip_id'], r['run'])) or tr_by.get((r['clip_id'], 1))
                tc = transcriber_cost(tr) if tr else None
                c = None if tc is None or c is None else c + tc
            costs.append(c)
        cost = f'${1000 * statistics.mean(costs):.2f}' if costs and None not in costs else 'unread (see PRICE_*)'
        split = sum(1 for r, v in pairs if v.get('slots_ok_across_calls') and not v.get('slots_ok'))
        by_run = {(r['clip_id'], r['run']): v for r, v in rows_all}
        call_key = lambda r: (by_run[(r['clip_id'], r['run'])]['function'], by_run[(r['clip_id'], r['run'])]['language'], by_run[(r['clip_id'], r['run'])].get('slots_ok'))
        cats = [v['categories'] for _, v in pairs if v.get('categories')]
        exp_c, given_c, right_c = (sum(x[k] for x in cats) for k in ('expected', 'given', 'right'))
        out.append(f'| {label} | {slots} | {recall} | {flips([r for r, _ in rows_all], call_key)} | {quant(lat, .95)} | {cost} | '
                   f'{"all in one" if not split else f"{split} split across calls"} | {f"{given_c}/{exp_c} / {right_c}/{given_c}" if exp_c else "–"} |')

    for ref_name, model in refs:
        ref = [(r, v) for r, v in judged.get(ref_name, []) if r['run'] == 1]
        if ref:
            out += ['', f'**{model} alone, on the reference transcripts:** all correct {pct([v["full"] for _, v in ref])}; '
                    f'{hard} {pct([v["full"] for r, v in ref if sub(clips[r["clip_id"]], "hard")])}; '
                    f'{second} {pct([v["full"] for r, v in ref if sub(clips[r["clip_id"]], "second")])}; '
                    f'code-switched {pct([v["full"] for r, v in ref if sub(clips[r["clip_id"]], "cs")])}; '
                    f'off-topic declined {pct([v["declined_ok"] for r, v in ref if clips[r["clip_id"]].expected_function == DECLINE and clips[r["clip_id"]].language])}.']

    if audio == 'tts':
        vendors = sorted({(r.get('voice') or {}).get('vendor') for rows in judged.values() for r, _ in rows if (r.get('voice') or {}).get('vendor')})
        if len(vendors) > 1:
            out += ['', "## By TTS vendor (a vendor's own voices can flatter its own transcriber)", '',
                    '| Candidate | ' + ' | '.join(f'All correct, {v} voices' for v in vendors) + ' |', '|---|' + '---|' * len(vendors)]
            for name, label in candidates:
                pairs = [(r, v) for r, v in judged.get(name, []) if r['run'] == 1 and r.get('voice')]
                if pairs:
                    out.append(f'| {label} | ' + ' | '.join(pct([v['full'] for r, v in pairs if r['voice'].get('vendor') == vd]) for vd in vendors) + ' |')

    if kept:
        out += ['', '## Listening pass (human-pass page, 0–2 a line)', '', '| Candidate | Lines scored | Points | At 0 (wrong or unusable) |', '|---|---|---|---|']
        for name, label in candidates:
            cells = {cid: cell for (cid, lab), cell in kept.items() if lab == label and score_value(cell) is not None}
            if cells:
                vals = [score_value(c) for c in cells.values()]
                zeros = sorted(cid for cid, c in cells.items() if score_value(c) == 0)
                out.append(f'| {label} | {len(vals)} | {sum(vals)}/{2 * len(vals)} | {", ".join(zeros) or "none"} |')

    if scores.get('errors'):
        out += ['', '## Provider errors', ''] + [f'- {n}: {len(e)} — ' + '; '.join(e[:3]) for n, e in scores['errors'].items()]

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f'scorecard{suffix}.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    (RESULTS / f'scores{suffix}.json').write_text(json.dumps(scores, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

    # The listening pass: every spoken line beside what was said, 0/1/2 to fill in. Summaries come first, since
    # nothing else checks what a summary says. Scores already written on the page are carried over.
    hp = [f'# Listening pass — {title}', '',
          'Score each spoken line 0/1/2: 0 = wrong or unusable, 1 = right but awkward, 2 = right and natural.', '']
    order = sorted(clips, key=lambda cid: clips[cid].expected_function != SUMMARY)
    for cid in order:
        c = clips[cid]
        rows = [(label, r, v) for name, label in candidates for r, v in judged.get(name, []) if r['clip_id'] == cid and r['run'] == 1]
        if not rows:
            continue
        hp += [f'### {cid} — {c.transcript or "(silence)"}', f'Expected: {c.expected_function}. {c.notes}', '', '| Candidate | Call | Spoken line | Score |', '|---|---|---|---|']
        for label, r, v in rows:
            hp.append(f'| {label} | {v["function"]} ({v.get("language")}) | {(v.get("line") or "—").replace("|", "/")} |{kept.get((cid, label), " ")}|')
        hp += [''] + [f'- {m}' for m in sorted({f'{label}: {m}' for label, r, v in rows for m in v.get('misses', [])})] + ['']
    hp_path.write_text('\n'.join(hp) + '\n', encoding='utf-8')
    if echo:
        print('\n'.join(out) + '\n')
    print(f'wrote results/scorecard{suffix}.md, results/human-pass{suffix}.md, results/scores{suffix}.json')
    return {'clips': clips, 'candidates': candidates, 'judged': judged, 'hard': hard, 'second': second}


def by_speaker(sets: dict[str, dict]) -> str:
    """All correct per candidate and speaker, then pooled over every speaker and run."""
    seen: list[tuple[str, str]] = []
    for s in sets.values():
        seen += [c for c in s['candidates'] if s['judged'].get(c[0]) and c not in seen]
    speakers = list(sets)
    any_set = next(iter(sets.values()))
    hard, second = any_set['hard'], any_set['second']
    out = ['# Scorecard draft — the recordings, by speaker', '',
           'All correct, run 1 / run 2, of the lines each speaker recorded; the pooled columns count every speaker and run.', '',
           '| Candidate | ' + ' | '.join(speakers) + f' | Pooled | Pooled, {hard} | Pooled, {second} | Pooled, code-switched |',
           '|---|' + '---|' * (len(speakers) + 4)]
    for name, label in seen:
        cells, pooled = [], []
        for sp in speakers:
            s = sets[sp]
            pairs = s['judged'].get(name) or []
            if not pairs:
                cells.append('–')
                continue
            runs = sorted({r['run'] for r, _ in pairs})
            cells.append(' / '.join(str(sum(v['full'] for r, v in pairs if r['run'] == k)) for k in runs) + f' of {len({r["clip_id"] for r, _ in pairs})}')
            pooled += [(s['clips'][r['clip_id']], v) for r, v in pairs]
        tot = lambda name_: pct([v['full'] for c, v in pooled if subset(c, name_, hard, second)])
        out.append(f'| {label} | ' + ' | '.join(cells) + f' | {tot("all")} | {tot("hard")} | {tot("second")} | {tot("cs")} |')
    return '\n'.join(out) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio', choices=AUDIO_SETS, default='tts')
    ap.add_argument('--speaker', help="with --audio real: one speaker's recordings; without it, every speaker and a table by speaker")
    ap.add_argument('--control', default='gemini', help="the run whose score defines good enough (default: gemini)")
    args = ap.parse_args()
    if args.audio == 'tts' or args.speaker:
        score_set(audio_set(args.audio, args.speaker), args.control)
        return
    speakers = real_speakers()
    if not speakers:
        raise SystemExit('no recordings yet: they go in clips/real/<speaker>/<file>')
    table = by_speaker({sp: score_set(f'real-{sp}', args.control, echo=False) for sp in speakers})
    (RESULTS / 'scorecard--real.md').write_text(table, encoding='utf-8')
    print('\n' + table)
    print(f'wrote {Path(RESULTS / "scorecard--real.md").relative_to(HERE)}')


if __name__ == '__main__':
    main()
