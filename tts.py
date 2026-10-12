"""Optional: speaks every manifest line with text-to-speech into clips/tts/<clip_id>.m4a.

    python tts.py --plan        who speaks what; no keys, no calls
    python tts.py               every line not yet spoken; --force speaks them again
    python tts.py --clips he01,en01

Generating a test set takes an hour where recording one takes a day. Read these two caveats before you
trust it:

1. A vendor's own voices can favour its own transcriber. When a candidate is also one of your TTS vendors,
   compare its score on its own voices with its score on the other vendor's (score.py does, per vendor).
   TTS measures the contract; real voices rank the candidates.
2. A voice may read its refusal instead of the line. Safety-sensitive lines are where it happens, and the
   clip then carries an apology, not the test. Transcribe every generated clip once and read the transcripts
   before anyone trusts a score.

Voices. Two vendors, so a transcriber's score can be read per vendor: OpenAI's TTS and Azure's neural voices.
Within each language the lines take turns, in manifest order, through Azure male, OpenAI female, Azure
female, OpenAI male. --openai-only uses OpenAI's voices alone.

Text. A line is spoken as the manifest's transcript writes it, or as its tts_text column, when that is
filled in: a code-switched line holds a word in another script, which one voice reads best spelt the way a
speaker of the line's language says it. The transcript, and so the scoring, keep the line as written. A
line with an empty transcript becomes five seconds of room tone, a silence test.

Sound. Like a phone recording: half a second before and after the speech, a faint pink-noise room tone
under it (no transcriber should meet digital silence), AAC at 48 kHz stereo, 128 kb/s, in .m4a.

clips/tts/voices.json records each clip's vendor, voice, gender and spoken text; run.py copies it into every
result line. Keys: OPENAI_API_KEY (with text-to-speech allowed), and AZURE_SPEECH_KEY with
AZURE_SPEECH_ENDPOINT unless --openai-only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import requests

from harness import TTS_CLIPS, load_manifest, require

OPENAI_MODEL = 'gpt-4o-mini-tts'
OPENAI_VOICES = {'f': 'coral', 'm': 'onyx'}
# Azure's voices for each language, female and male, from its language-support page; checked against the live
# voice list before anything is spoken.
AZURE_VOICES = {
    'he': ('he-IL-HilaNeural', 'he-IL-AvriNeural'),
    'en': ('en-US-AvaNeural', 'en-US-AndrewNeural'),
    'ar': ('ar-SA-ZariyahNeural', 'ar-SA-HamedNeural'),
    'es': ('es-ES-ElviraNeural', 'es-ES-AlvaroNeural'),
    'fr': ('fr-FR-DeniseNeural', 'fr-FR-HenriNeural'),
    'pt': ('pt-BR-FranciscaNeural', 'pt-BR-AntonioNeural'),
    'ru': ('ru-RU-SvetlanaNeural', 'ru-RU-DmitryNeural'),
    'uk': ('uk-UA-PolinaNeural', 'uk-UA-OstapNeural'),
    'zh': ('zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural'),
}
ROTATION = [('azure', 'm'), ('openai', 'f'), ('azure', 'f'), ('openai', 'm')]
ROOM_TONE = 'anoisesrc=color=pink:amplitude=0.002:sample_rate=48000'
PAD_MS = 500


def azure_region() -> str:
    explicit = os.environ.get('AZURE_SPEECH_REGION')
    if explicit:
        return explicit
    m = re.match(r'https://([a-z0-9]+)\.api\.cognitive\.microsoft\.com', os.environ['AZURE_SPEECH_ENDPOINT'])
    if not m:
        raise SystemExit('Set AZURE_SPEECH_REGION too: the endpoint does not name its region.')
    return m.group(1)


def plan(clips, openai_only: bool) -> dict[str, dict]:
    """Each line's vendor, voice, gender and spoken text, by manifest order within its language."""
    turns: dict[str, int] = {}
    out = {}
    for c in clips:
        if not c.transcript.strip():
            out[c.id] = {'vendor': None, 'voice': None, 'gender': None, 'text': ''}
            continue
        vendor, gender = ROTATION[turns.get(c.language, 0) % len(ROTATION)]
        turns[c.language] = turns.get(c.language, 0) + 1
        if openai_only or c.language not in AZURE_VOICES:
            vendor = 'openai'
        voice = AZURE_VOICES[c.language][0 if gender == 'f' else 1] if vendor == 'azure' else OPENAI_VOICES[gender]
        out[c.id] = {'vendor': vendor, 'voice': voice, 'gender': gender, 'text': c.tts_text or c.transcript}
    return out


class Refused(Exception):
    """A TTS service would not speak a line; the other lines go on."""


def speak(vendor: str, voice: str, text: str) -> bytes:
    """WAV bytes of the text in that voice."""
    if vendor == 'openai':
        r = requests.post('https://api.openai.com/v1/audio/speech',
                          headers={'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'},
                          json={'model': OPENAI_MODEL, 'voice': voice, 'input': text, 'response_format': 'wav'}, timeout=60)
    else:
        locale = '-'.join(voice.split('-')[:2])
        ssml = f"<speak version='1.0' xml:lang='{locale}'><voice name='{voice}'>{escape(text)}</voice></speak>"
        r = requests.post(f'https://{azure_region()}.tts.speech.microsoft.com/cognitiveservices/v1',
                          headers={'Ocp-Apim-Subscription-Key': os.environ['AZURE_SPEECH_KEY'],
                                   'Content-Type': 'application/ssml+xml',
                                   'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm',
                                   'User-Agent': 'voice-provider-harness'},
                          data=ssml.encode('utf-8'), timeout=60)
    if not r.ok:
        raise Refused(f'{vendor} TTS refused {voice}: {r.status_code} {r.text[:300]}')
    if not r.content.startswith(b'RIFF'):
        raise SystemExit(f'{vendor} TTS answered {voice} with something that is not WAV ({r.headers.get("Content-Type")})')
    return r.content


def azure_voice_list() -> dict[str, str]:
    """ShortName → Gender, from Azure's live list."""
    r = requests.get(f'https://{azure_region()}.tts.speech.microsoft.com/cognitiveservices/voices/list',
                     headers={'Ocp-Apim-Subscription-Key': os.environ['AZURE_SPEECH_KEY']}, timeout=30)
    r.raise_for_status()
    return {v['ShortName']: v.get('Gender', '') for v in r.json()}


def render(speech: Path | None, out: Path) -> None:
    """Pad, lay on room tone, encode as a phone does; no speech gives five seconds of room tone."""
    cmd = ['ffmpeg', '-v', 'error', '-y']
    if speech is None:
        cmd += ['-f', 'lavfi', '-i', f'{ROOM_TONE}:duration=5', '-filter_complex', '[0:a]anull[out]']
    else:
        pad = f'aresample=48000,adelay=delays={PAD_MS}:all=1,apad=pad_dur={PAD_MS / 1000}'
        cmd += ['-i', str(speech), '-f', 'lavfi', '-i', ROOM_TONE,
                '-filter_complex', f'[0:a]{pad}[s];[s][1:a]amix=inputs=2:duration=first:normalize=0[out]']
    cmd += ['-map', '[out]', '-ar', '48000', '-ac', '2', '-c:a', 'aac', '-b:a', '128k', str(out)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', action='store_true', help='print who speaks what, and stop')
    ap.add_argument('--clips', help='comma-separated clip ids')
    ap.add_argument('--force', action='store_true', help='speak lines again that already have a clip')
    ap.add_argument('--openai-only', action='store_true', help="every line in an OpenAI voice")
    args = ap.parse_args()

    clips = load_manifest('tts')
    voices = plan(clips, args.openai_only)
    if args.clips:
        clips = [c for c in clips if c.id in args.clips.split(',')]
    if args.plan:
        for c in clips:
            v = voices[c.id]
            print(f'{c.id:<6} {c.language or "-":<3} {v["vendor"] or "-":<7} {v["voice"] or "room tone":<22} {v["gender"] or "-"}  {v["text"]}')
        return

    if args.openai_only:
        require('OPENAI_API_KEY')
        listed = {}
    else:
        require('OPENAI_API_KEY', 'AZURE_SPEECH_KEY', 'AZURE_SPEECH_ENDPOINT')
        listed = azure_voice_list()
        missing = sorted({v['voice'] for v in voices.values() if v['vendor'] == 'azure'} - listed.keys())
        if missing:
            raise SystemExit(f'Azure lists no voice named {", ".join(missing)} in {azure_region()}; fix AZURE_VOICES first')

    TTS_CLIPS.mkdir(parents=True, exist_ok=True)
    log_path = TTS_CLIPS / 'voices.json'
    log = json.loads(log_path.read_text(encoding='utf-8')) if log_path.exists() else {}
    with tempfile.TemporaryDirectory(dir=TTS_CLIPS) as tmp:
        for c in clips:
            if c.path.exists() and not args.force:
                continue
            v = dict(voices[c.id])
            if v['vendor'] == 'azure':
                v['gender'] = {'Female': 'f', 'Male': 'm'}.get(listed.get(v['voice'], ''), v['gender'])
            speech = None
            if v['vendor']:
                speech = Path(tmp) / f'{c.id}.wav'
                try:
                    speech.write_bytes(speak(v['vendor'], v['voice'], v['text']))
                except Refused as e:
                    print(f'{c.id:<6} REFUSED, no clip: {e}')
                    continue
            render(speech, c.path)
            log[c.id] = v
            log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'{c.id:<6} {v["vendor"] or "-":<7} {v["voice"] or "room tone":<22} {c.path.name}')
    print(f'{len(log)} clips in clips/tts/; transcribe them once and read the transcripts before trusting a score')


if __name__ == '__main__':
    main()
