"""Runs one provider over the clips and writes results/<name>.jsonl, one line per clip and run.

    python run.py --check                                   the manifest against clips/, and which keys this shell has
    python run.py --provider soniox --repeat 2              every clip that exists, twice (the flip rate needs two runs)
    python run.py --provider gemini --clips he01,en01       two clips
    python run.py --provider llm_text --model <id>          a text model on the manifest's own transcripts
    python run.py --provider llm_text --model <id> --source soniox
                                                            the same text model on a transcriber's transcripts
    python run.py --provider soniox --audio real --speaker anna
                                                            one speaker's recordings, clips/real/anna/
    python run.py --show soniox                             print a results file again
    python run.py --leftovers soniox [--delete]             what the provider's account still holds

A line holds clip_id, run, language_detected, transcript or function_calls, latency_ms, usage and the raw
answer, or an error when the provider failed. The file is rewritten on every run unless --merge is given;
a run that fails or is stopped never overwrites the last good results.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ModuleNotFoundError as e:
    raise SystemExit(f'{e.name} is not installed: pip install -r requirements.txt') from None

from harness import AUDIO_SETS, HERE, MANIFEST_FILE, RESULTS, AdapterError, Clip, DailyLimit, audio_set, load_manifest, probe, read_results, require, results_path, tts_voices, variant

PROVIDERS = sorted(p.stem for p in (HERE / 'adapters').glob('*.py') if not p.stem.startswith('_'))


def adapter_module(name: str):
    """The provider's adapter; a package missing from a partial install stops the run with the line that installs
    it, not a stack trace."""
    try:
        return importlib.import_module(f'adapters.{name}')
    except ModuleNotFoundError as e:
        raise SystemExit(f'adapter {name} needs {e.name}, which is not installed: pip install -r requirements.txt') from None


def select(clips: list[Clip], args, needs_audio: bool = True) -> list[Clip]:
    if args.clips:
        wanted = args.clips.split(',')
        unknown = [c for c in wanted if c not in {x.id for x in clips}]
        if unknown:
            raise SystemExit(f'not in {MANIFEST_FILE.name}: {", ".join(unknown)}')
        chosen = [c for c in clips if c.id in wanted]
    else:
        chosen = clips
    if args.skip:
        skip = set(args.skip.split(','))
        chosen = [c for c in chosen if not (c.tags & skip)]
    if args.only:
        only = set(args.only.split(','))
        chosen = [c for c in chosen if c.tags & only]
    if not needs_audio:
        return part(chosen, args)
    missing = [c.id for c in chosen if not c.exists]
    if missing and args.clips:
        where = 'clips/tts/ (run tts.py, or record them)' if args.audio == 'tts' else f'clips/real/{args.speaker}/'
        raise SystemExit(f'no {args.audio} clip yet for: {", ".join(missing)} (expected in {where})')
    if missing:
        print(f'skipping {len(missing)} clips not yet recorded: {few(missing)}')
    return part([c for c in chosen if c.exists], args)


def part(chosen: list[Clip], args) -> list[Clip]:
    """--part k/n: the k-th of n even slices of the chosen clips, in manifest order, so a pass that shares a daily
    budget can run over several days; with --merge each day's lines join the same results file."""
    if not getattr(args, 'part', None):
        return chosen
    k, n = (int(x) for x in args.part.split('/'))
    if not 1 <= k <= n:
        raise SystemExit('--part k/n: k from 1 to n')
    size = -(-len(chosen) // n)
    return chosen[(k - 1) * size:k * size]


def few(ids: list[str]) -> str:
    return ', '.join(ids) if len(ids) <= 12 else ', '.join(ids[:12]) + f' and {len(ids) - 12} more'


def show_line(r: dict) -> str:
    if r.get('error'):
        kept = [k for k, ok in (r.get('deleted') or {}).items() if not ok]
        return f'{r["clip_id"]:<6} run {r["run"]}  ERROR {r["error"]}' + (f'  (NOT deleted at the provider: {", ".join(kept)})' if kept else '')
    lang = r.get('language_detected')
    mark = '✓' if lang == r['language_expected'] else '✗'
    timing = ', '.join(f'{k} {v}' for k, v in (r.get('timing_ms') or {}).items())
    head = f'{r["clip_id"]:<6} run {r["run"]}  {r["language_expected"] or "-"}→{lang} {mark}  {r["latency_ms"]} ms' + (f' ({timing})' if timing else '')
    lines = [head]
    if r.get('transcript') is not None:
        lines += [f'       ref  {r["transcript_ref"]}', f'       hyp  {r["transcript"]}']
    else:
        lines.append(f'       said {r["transcript_ref"]}')
        if r.get('source') not in (None, 'reference'):
            lines.append(f'       heard {r.get("source_transcript")}  ({r["source"]}; pipeline {r.get("pipeline_latency_ms")} ms)')
    for call in r.get('function_calls') or []:
        lines.append(f'       call {call["name"]} {json.dumps(call["args"], ensure_ascii=False)}')
    return '\n'.join(lines)


def check(audio: str) -> None:
    clips = load_manifest(audio)
    print(f'audio set: {audio}; manifest: {MANIFEST_FILE.relative_to(HERE)}')
    keys, missing = {'OPENAI_API_KEY'}, []
    for p in PROVIDERS:
        try:
            keys |= set(importlib.import_module(f'adapters.{p}').KEYS)
        except ModuleNotFoundError as e:
            missing.append(f'{p} needs {e.name}')
    print(f'{len(clips)} lines; keys in this shell: ' + ', '.join(f'{k} {"set" if os.environ.get(k) else "not set"}' for k in sorted(keys)))
    if missing:
        print(f'not installed: {"; ".join(missing)}. Run: pip install -r requirements.txt')
    recorded = [c for c in clips if c.exists]
    for c in recorded:
        p = probe(c.path)
        print(f'  {c.id:<6} {p["seconds"]:5.1f} s  {p["codec"]} {p["sample_rate"]} Hz x{p["channels"]}')
    waiting = [c.id for c in clips if not c.exists]
    print(f'recorded {len(recorded)}/{len(clips)}' + (f'; still to record: {few(waiting)}' if waiting else ''))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', choices=PROVIDERS)
    ap.add_argument('--clips', help='comma-separated clip ids')
    ap.add_argument('--repeat', type=int, default=1, help='runs over the set; 2 gives the flip rate')
    ap.add_argument('--hints', type=lambda s: s.split(','), help='language hints for transcribers that take them, e.g. he,en; '
                                                                 'off by default, so detection is measured')
    ap.add_argument('--vocab', choices=('names', 'full'), help="prime the transcriber: the context's names in both scripts, "
                                                               'or those plus lexicon.txt; results carry +names or +full')
    ap.add_argument('--vocab-field', choices=('keywords', 'prompt'), help='openai_stt: where the vocabulary goes (default keywords)')
    ap.add_argument('--check', action='store_true', help='report the clips found, their format, and which keys this shell has')
    ap.add_argument('--show', metavar='RESULTS', help='print results/<name>.jsonl again')
    ap.add_argument('--leftovers', metavar='PROVIDER', choices=PROVIDERS, help="list what the provider's account still holds")
    ap.add_argument('--delete', action='store_true', help='with --leftovers: delete the copies named after a clip in the manifest')
    ap.add_argument('--source', default='reference', help="text steps: reference (the manifest's transcripts) or a transcriber whose results to read")
    ap.add_argument('--model', help='the model to ask; required by llm_text and llm_claude, optional elsewhere')
    ap.add_argument('--base-url', help='llm_text, openai_stt: an OpenAI-compatible endpoint other than OpenAI\'s')
    ap.add_argument('--key-env', help='llm_text, openai_stt: the environment variable holding that endpoint\'s key')
    ap.add_argument('--response-format', help='openai_stt: json (default) or verbose_json, for Whisper-style endpoints that name the language')
    ap.add_argument('--reasoning-effort', help='llm_text: sent as reasoning_effort, for a model that reasons by default')
    ap.add_argument('--builder', metavar='EXPORT', help="llm_text: ask your server's own export for the text request (README, The bridge)")
    ap.add_argument('--skip', metavar='TAGS', help='leave out clips with these tags, e.g. other')
    ap.add_argument('--only', metavar='TAGS', help='keep only clips with one of these tags, e.g. codeswitch')
    ap.add_argument('--pace', choices=('fast', 'live'), default='fast', help='soniox_rt: send each clip whole, or at real-time speed')
    ap.add_argument('--audio', choices=AUDIO_SETS, default='tts', help="tts (default): clips/tts/; real: one speaker's recordings (with --speaker)")
    ap.add_argument('--speaker', help='with --audio real: the folder under clips/real/; results are named <name>--real-<speaker>')
    ap.add_argument('--merge', action='store_true', help="replace only these clips' lines in the results file and keep the rest")
    ap.add_argument('--part', metavar='K/N', help='the k-th of n even slices of the chosen clips (with --merge, a pass over several days)')
    args = ap.parse_args()
    args.audio = audio_set(args.audio, args.speaker)  # 'tts' or 'real-<speaker>' from here on

    if args.check:
        return check(args.audio)
    if args.show:
        for r in read_results(args.show):
            print(show_line(r))
        return
    if args.leftovers:
        adapter = adapter_module(args.leftovers)
        if not hasattr(adapter, 'leftovers'):
            print(f"{adapter.NAME}: nothing to list or delete through the API; what the provider keeps is in the adapter's docstring")
            return
        require(*adapter.KEYS)
        lines = adapter.leftovers(args.delete)
        print('\n'.join(lines) if lines else f'{adapter.NAME}: the account holds nothing')
        return
    if not args.provider:
        ap.error('--provider, --check, --show or --leftovers')

    adapter = adapter_module(args.provider)
    require(*adapter.KEYS)
    if hasattr(adapter, 'prepare'):
        adapter.prepare(args)
    needs_audio = adapter.needs_audio(args) if hasattr(adapter, 'needs_audio') else True
    clips = select(load_manifest(args.audio), args, needs_audio)
    voices = tts_voices() if args.audio == 'tts' else {}
    RESULTS.mkdir(exist_ok=True)
    # A run on the reference transcripts uses no audio, so its name carries no audio set.
    out = results_path(adapter.output_name(args) if hasattr(adapter, 'output_name') else adapter.NAME + variant(args), args.audio if needs_audio else None)
    # Written beside the results and moved over them only when the run completes. Three identical HTTP errors on
    # the first three clips (no credit, a bad key, a model not enabled) stop the run rather than repeating it.
    partial = out.with_name(out.name + '.partial')
    opening: list[str | None] = []
    stop = None
    rest: list[str] = []
    with partial.open('w', encoding='utf-8') as f:
        for run in range(1, args.repeat + 1):
            if stop or rest:
                break
            args.run = run
            for clip in clips:
                base = {
                    'provider': adapter.NAME, 'clip_id': clip.id, 'run': run,
                    'at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'duration_s': probe(clip.path)['seconds'] if clip.exists else None,
                    'language_expected': clip.language, 'transcript_ref': clip.transcript,
                    'audio': args.audio if needs_audio else None,
                    'voice': voices.get(clip.id) if needs_audio else None,
                }
                try:
                    r = {**base, **adapter.run(clip, args)}
                    r['function_calls'] = r.get('function_calls') or ([r['function_call']] if r.get('function_call') else [])
                except DailyLimit as e:
                    # A limit that will not clear soon: stop at once, keep what ran, and say what is left.
                    print(f'{clip.id:<6} run {run}  STOPPED: {e}')
                    rest = [c.id for c in clips[clips.index(clip):]]
                    break
                except (AdapterError, requests.RequestException, KeyError, ValueError) as e:
                    detail = ''
                    if isinstance(e, requests.HTTPError) and e.response is not None:
                        detail = f' {e.response.status_code}: {e.response.text[:300]}'
                    r = {**base, 'model': getattr(adapter, 'MODEL', None), 'error': f'{type(e).__name__}{detail or ": " + str(e)}',
                         'deleted': getattr(e, 'deleted', None)}
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
                print(show_line(r))
                if run == 1 and len(opening) < 3:
                    opening.append(r.get('error'))
                    kinds = {(e or '').split(':')[0] for e in opening}
                    if len(opening) == 3 and all(opening) and len(kinds) == 1 and opening[0].startswith('HTTPError'):
                        stop = f'stopped: the first three clips failed the same way ({opening[0][:200]}); the last good results are untouched'
                        break
    if stop:
        # The failed lines are kept beside the results as the record of why: a refusal is a finding too.
        partial.replace(out.with_name(out.name + '.stopped'))
        raise SystemExit(f'{stop}; the failed lines are in {out.name}.stopped')
    if args.merge and out.exists():
        ran = {c.id for c in clips} - set(rest)
        kept = [line for line in out.read_text(encoding='utf-8').splitlines() if line.strip() and json.loads(line)['clip_id'] not in ran]
        partial.write_text(''.join(f'{line}\n' for line in kept) + partial.read_text(encoding='utf-8'), encoding='utf-8')
    partial.replace(out)
    out.with_name(out.name + '.stopped').unlink(missing_ok=True)
    print(f'wrote {Path(out).relative_to(HERE)}')
    if rest:
        raise SystemExit(f'stopped at a long rate limit with {len(rest)} clips not run; once it has cleared, the same command '
                         f'with --merge --clips {",".join(rest)} finishes the run')


if __name__ == '__main__':
    sys.exit(main())
