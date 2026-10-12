/**
 * The request your server sends, written down for the harness: the one rule is that the harness sends what
 * production sends.
 *
 *   node dump-request.mjs [module] [export]     writes request-template.json
 *   node dump-request.mjs [module] --serve      answers one JSON line per request, for harness.Bridge
 *
 * module defaults to examples/request-builder.example.ts and export to buildRequest. Point them at your
 * server's own request builder, imported where it stands, so nothing of the prompt is copied or retyped.
 * The export is called with the context (context.json at the top of the harness, or the example's) and an
 * audio part whose data is a placeholder, and returns { model, body }, the placeholder somewhere in body.
 * Any other field it returns is kept in the template too.
 *
 * The template records the SHA-256 of JSON.stringify(body), the bytes your server would send; harness.py
 * proves Python serialises the body to the same bytes before an adapter puts a clip where the placeholder is.
 *
 * Node 23.6 and later import a .ts module by stripping its types. For older Node, or TypeScript that needs a
 * compiler (enums, path aliases), run it as  npx tsx dump-request.mjs …  instead.
 *
 * --serve reads lines like {"op":"call","export":"textRequest","args":[context, "transcript", "he"]} and
 * answers {"result": …}, awaiting a promise; {"op":"exports"} answers the module's export names. Errors come
 * back as {"error": "…"}. This is how an adapter can ask your server for a request live.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

const here = import.meta.dirname;
const args = process.argv.slice(2);
const serve = args.includes('--serve');
const positional = args.filter(a => !a.startsWith('--'));
const modulePath = resolve(positional[0] ?? resolve(here, 'examples/request-builder.example.ts'));
const exportName = positional[1] ?? 'buildRequest';
const PLACEHOLDER = '__AUDIO_BASE64__';

const mod = await import(pathToFileURL(modulePath).href);

if (serve) await answer();
else await template();

function context() {
  const own = resolve(here, 'context.json');
  const file = existsSync(own) ? own : resolve(here, 'examples/context.example.json');
  return JSON.parse(readFileSync(file, 'utf8'));
}

async function template() {
  const build = mod[exportName];
  if (typeof build !== 'function') throw new Error(`${modulePath} has no function export named ${exportName}`);
  const built = await build(context(), { mimeType: 'audio/mp4', data: PLACEHOLDER });
  if (!built || typeof built.model !== 'string' || typeof built.body !== 'object') {
    throw new Error(`${exportName} must return { model, body }`);
  }
  const text = JSON.stringify(built.body);
  if (text.split(JSON.stringify(PLACEHOLDER)).length !== 2) {
    throw new Error(`${exportName}'s body must hold the audio placeholder exactly once`);
  }
  const { model, body, ...extra } = built;
  const out = {
    generatedBy: 'dump-request.mjs; regenerate it, never edit it',
    model,
    audioPlaceholder: PLACEHOLDER,
    bodySha256: createHash('sha256').update(text, 'utf8').digest('hex'),
    ...extra,
    body,
  };
  writeFileSync(resolve(here, 'request-template.json'), `${JSON.stringify(out, null, 2)}\n`);
  console.log(`request-template.json: ${model}, sha256 ${out.bodySha256.slice(0, 12)}`);
}

async function answer() {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    let out;
    try {
      const req = JSON.parse(line.replace(/^﻿/, '')); // a PowerShell pipe opens with a byte-order mark
      if (req.op === 'exports') out = { exports: Object.keys(mod) };
      else if (req.op === 'call') {
        if (typeof mod[req.export] !== 'function') throw new Error(`no function export named ${req.export}`);
        out = { result: await mod[req.export](...(req.args ?? [])) };
      } else out = { error: `unknown op ${req.op}` };
    } catch (err) {
      out = { error: String(err) };
    }
    process.stdout.write(`${JSON.stringify(out)}\n`);
  }
}
