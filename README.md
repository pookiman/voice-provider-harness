# voice-provider-harness

A small harness for choosing a voice provider by measuring **your contract**, not the vendor: the request your
server already sends, the function call it expects back, scored the way your app would apply it. It
accompanies the article [Measure the contract, not the vendor](ARTICLE_URL).

**The one rule: the harness sends what production sends.** Your server's request builder is imported where it
stands and written to `request-template.json`, and every adapter starts from that file:
- **The control,** today's provider, gets it byte for byte, and a hash proves it.
- **Audio-native candidates** get the same body with the host changed.
- **Transcriber-plus-text-model candidates** get the same instruction and declarations, translated
  mechanically. The audio part becomes one sentence: "The user said (transcribed; the transcriber heard
  language he): «…»".

Nothing is rephrased for a vendor.

```
clips/*.m4a ──► run.py --provider X ──► results/X.jsonl ──► score.py ──► scorecard
                      ▲
   your request builder ──► dump-request.mjs ──► request-template.json
```

No results are included, and none of the article's figures are reproduced here. The article's numbers are a
dated snapshot; vendors change models and prices under you.

## Quick start

You need Python 3.11+, ffmpeg (with ffprobe) on the PATH, and Node 23.6+, which runs a TypeScript builder
directly; on older Node, use `npx tsx` wherever `node` appears below.

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # .venv/bin/python on macOS and Linux
node dump-request.mjs                                          # writes request-template.json from the example builder
.venv/Scripts/python run.py --check                            # the manifest, the clips found, the keys set
```

The repository ships an example contract, manifest and context in `examples/`, for a community garden's shared
list. You can run everything against them before pointing the harness at your own server. There are no
recordings: make your own clips (`tts.py`, or record them) into `clips/tts/<clip_id>.m4a`.

**Keys** come from your shell's environment, one variable per provider, set for the session and never written
to a file. `.env.example` lists the names only.

| Provider | Variables |
|---|---|
| `gemini` | `GEMINI_API_KEY` |
| `openai_stt`, `llm_text` | `OPENAI_API_KEY`, or any variable you name with `--key-env` |
| `soniox`, `soniox_rt` | `SONIOX_API_KEY` |
| `azure_fast` | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_ENDPOINT` |
| `llm_claude` | `ANTHROPIC_API_KEY` |

Then run each candidate twice and score:

```
.venv/Scripts/python run.py --provider gemini --repeat 2
.venv/Scripts/python run.py --provider soniox --repeat 2
.venv/Scripts/python run.py --provider llm_text --model <model id> --source soniox --repeat 2
.venv/Scripts/python score.py
```

`score.py` writes `results/scorecard.md`, `results/human-pass.md` (every spoken line, for a listener) and
`results/scores.json` (every verdict, so any number traces to its clips). Real recordings go in
`clips/real/<speaker>/` and run with `--audio real --speaker <speaker>`; `score.py --audio real` adds a table
by speaker.

## Plugging in your own contract

`dump-request.mjs` imports a module and calls one of its exports:

```
node dump-request.mjs path/to/your/server/requestBuilder.ts buildRequest
```

The export is called as `buildRequest(context, { mimeType: 'audio/mp4', data: '__AUDIO_BASE64__' })` and must
return `{ model, body }`, the placeholder appearing exactly once in `body`. The context is `context.json` at
the top of the harness, or the example's. The template written is:

```json
{ "generatedBy": "…", "model": "…", "audioPlaceholder": "__AUDIO_BASE64__", "bodySha256": "…", "body": { … } }
```

`bodySha256` is taken over `JSON.stringify(body)`, the bytes your server would send. `harness.py` refuses a
template whose body Python does not serialise to the same bytes, so the template is regenerated, never edited.
Keep `request-template.json` out of git (the ignore file does), so it cannot drift from the code.

The adapters expect a Gemini-shaped body (`systemInstruction`, `tools[0].functionDeclarations`, `toolConfig`,
`generationConfig`, a user turn of `[audio, text]`), as `examples/request-builder.example.ts` builds. If your
server speaks another API, either have your builder return that shape for the harness, or adapt `gemini.py` and
`llm_text.translated()`, the two places that read it.

**The bridge.** For a text step your server builds itself, `run.py --provider llm_text --builder textRequest`
asks your module's export for the request, live, through `dump-request.mjs <module> --serve` (`harness.Bridge`).
The export is called as `textRequest(context, transcript, heardLanguage, model)` and returns the Chat Completions
body your server would send. `HARNESS_MODULE` names the module and `HARNESS_NODE` the command that runs it.

## The manifest

`manifest.csv` at the top of the harness, or `examples/manifest.example.csv`. One row per test line:

| Column | Meaning |
|---|---|
| `clip_id` | A short id; the TTS clip is `clips/tts/<clip_id>.m4a` |
| `file` | The recording's file name, looked up in `clips/real/<speaker>/` for real voices |
| `speaker` | A label for who reads it (A, B…), for planning recordings |
| `language` | The ISO 639-1 code the person speaks; empty for a clip with no speech |
| `synthetic` | 1 for a line written to be generated rather than recorded; informational |
| `transcript` | What is said, verbatim; the reference for WER, recall and the text-only runs |
| `expected_function` | One of the contract's functions |
| `expected_slots` | JSON: what a user would accept (below) |
| `notes` | One-word tags first, separated by `;`, then prose: `codeswitch`, `edge`, `other`, `ambiguous`, `off-topic`… |
| `tts_text` | Optional: the text a TTS voice should read, when it differs from `transcript` (a word in another script) |

`expected_slots` holds the shapes the judge knows:
- `items` with `name`, `quantity`, `category`;
- `tasks` with `title`, `assigned_to`, `due_date`;
- `events` with `title`, `date`, `start_time`, `duration_minutes`, `attendees`;
- `task` (the pending title a completion must find);
- `_words`, the content words for recall.

Any accepted value can be a list of alternatives: `"title": ["prune the apple trees", "prune apple trees"]`. A
content word can be a list of its spellings in two scripts: `["Shira", "שירה"]`.

`context.json` is the frozen data every request is built from. It holds `today` and `weekday`, so "next
Thursday" resolves the same for every provider, and:
- `speaker` and `members`, with optional `aliases` (other-script spellings, used for priming and for the names table);
- `languages`: the hard language first, then the other main one;
- `tasks`, `items` and `events`.

Use invented people and invented data only.

## Scoring

**Gates first; any miss removes the candidate:**
- the language identified on clean clips (≥ 95%) and on every code-switched one;
- the right function name (≥ 95%);
- every off-topic and override line declined, and no real command declined;
- the spoken line written entirely in the declared language's script. This is a Unicode-block test, not a
  model's opinion;
- all-correct within two points of the control, on the hard language and on code-switched lines.

**Then the ranking,** in the order it decides:
- slot accuracy on the hard language;
- content-word recall;
- flip rate;
- latency at the 95th percentile, end to end;
- cost per thousand commands, from the vendors' own usage fields and the prices in `score.py`'s `PRICE_*`
  tables, which you must read off the vendors' pages yourself;
- whether every entry landed in one call.

- **Content-word recall** is the share of the manifest's `_words` (item names, people, days, numbers) that
  survive in a transcript. It is more honest than WER for this kind of feature: a transcript can have a high
  WER on particles and still carry every word the app needs, and "2" for "two" is an error to WER and right
  to recall.
- **The normaliser** (`normalise.py`) is applied to both sides of every comparison. It lowers case, drops
  Hebrew points and punctuation, turns dashes into spaces, and collapses whitespace. It deliberately folds
  nothing else.
- **Wrong writes** are counted beside all-correct: a miss whose calls would still add or change something the
  person did not ask for. Two misses can look equal to "all slots correct" and be very different to a user.
- **Everything runs twice** (`--repeat 2`), and the flip rate, the share of clips whose answer changed
  between runs, is reported as a result. The same model on the same input can change a few verdicts in a
  hundred; with ten code-switched clips, one clip is ten points. A difference smaller than the flip rate is
  weather.

**Text-to-speech clips measure the contract; real voices rank the candidates.** A vendor's own TTS voices can
flatter its own transcriber (`score.py` reports results per TTS vendor), and a voice can read its refusal
instead of a line. Transcribe a generated set once and read it before trusting it, and finish with a few
dozen real recordings from more than one speaker.

## Containment

- **Keys** live in the shell's environment and in no file at all; the scripts read the environment only.
- **Invented data only:** people, tasks, events and items. Several vendors keep uploads by default, and the
  clips go to all of them.
- **Delete what vendors keep.** The Soniox adapter deletes its uploaded file and its transcription after
  every clip, and `run.py --leftovers soniox --delete` lists and removes anything left. Read each vendor's
  retention terms, and revoke every key the day you are done.
- **Keep the harness out of the product.** If you copy it into a monorepo as a sibling folder, add that
  folder to three ignore lists in one commit, with a comment saying why:
  - the TypeScript compiler's `exclude` (a base config's `allowJs` will otherwise compile the `.mjs` script);
  - the linter's ignore list;
  - your build or deploy tool's own ignore file, which decides what is uploaded and is not `.gitignore`.
- **Recordings and results never enter git.** `clips/`, `results/` and the generated template are ignored.

## Provider notes

| Adapter | Kind | Takes the phone's m4a | Synchronous | Vocabulary field |
|---|---|---|---|---|
| `gemini` | audio-native | yes, inline | yes | none; the prompt is the template's |
| `openai_stt` | transcriber, any OpenAI-compatible endpoint | yes | yes | `keywords[]`, or `prompt` with `--vocab-field prompt` (Whisper-style) |
| `soniox` | transcriber | yes | no: upload, create, poll, fetch, about two to three seconds flat | `context.terms`, with a domain line |
| `soniox_rt` | transcriber, streaming | no: repackaged as ADTS by ffmpeg first | a WebSocket; time grows with the clip's length | `context.terms` |
| `azure_fast` | transcriber | yes | yes | `phraseList`; candidate locales from the context's `languages` |
| `llm_text` | text step, any OpenAI-compatible endpoint | – | yes | – |
| `llm_claude` | text step, Anthropic | – | yes | – |

`--vocab names` primes a transcriber with the context's names in both scripts; `--vocab full` adds the terms in
`lexicon.txt`, one per line. Names are the cheap, safe win. A lexicon that holds your test set's own words is a
ceiling, not a result, and it can put words in a transcript that were never said.

Asynchronous APIs cost a flat wait that model latency tables do not show. Time the pipeline end to end, from
where your server will stand, and read the laps each adapter records.

## Notice

Vendor and product names belong to their owners; nothing here is an endorsement or an affiliation. MIT
licensed; see `LICENSE`.
