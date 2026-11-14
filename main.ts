/**
 * free-model-aggregator-01 — one self-contained server.
 *
 * Serves the [OI] Chat Completions API and the Anthropic Messages API (JSON and
 * server-sent events), dials ten simulated free-tier providers over their own
 * wires, and answers every request with the highest-rated free model that can
 * carry it.
 *
 * Design notes
 *  - Everything the trial tells us arrives in $MAKEFUN_PROMPT_FILE: the provider
 *    directory, the quality table (the sealed truth: rating, context length and
 *    capabilities of every model, free or paid) and the workload manifest. We
 *    parse it, probe each provider's own catalog route, and register one catalog.
 *  - A request is passed through to the chosen provider in that provider's wire
 *    so the provider's echo of system/user/tools/max_tokens/stop/image/json
 *    matches what the client sent, and the provider's answer is returned in the
 *    wire the client used, with the provider's own usage counts and the stop
 *    reason mapped into the client's vocabulary.
 *  - Routing is a deterministic greedy plan over the remaining requests that
 *    spreads load across providers by their observed success/failure counts, so
 *    no single free-tier budget runs dry before the workload ends. Every refusal
 *    (402/429/5xx) is recorded and the next best model on another provider is
 *    called; a provider is demoted for the rest of the trial once it refuses.
 *  - Standard library only, no wall clock, no randomness: the same brief and the
 *    same provider answers always produce the same state export.
 */

import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');

const asNumber = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

const asBool = (v: unknown): boolean => v === true;

const sha8 = (text: string): string =>
  createHash('sha256').update(text).digest('hex').slice(0, 8);

const sha8bytes = (bytes: Buffer): string =>
  createHash('sha256').update(bytes).digest('hex').slice(0, 8);

const clampText = (v: unknown): string => (typeof v === 'string' ? v : '');

/** Fixed timestamp: the export must not depend on the wall clock. */
const CREATED = 1767225600;
const CREATED_AT = '2026-01-01T00:00:00Z';

const log = (msg: string): void => {
  process.stderr.write(`[aggregator] ${msg}\n`);
};

// ---------------------------------------------------------------------------
// brief parsing: three JSON documents inside prose
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  path: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requires {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

interface RequestSpec {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requires;
}

interface Brief {
  providers: Provider[];
  ratings: RatingRow[];
  manifest: RequestSpec[];
}

const EMPTY_REQUIRES: Requires = {
  tools: false,
  vision: false,
  json: false,
  minContext: null,
};

/** Every balanced JSON object in a text, in order, skipping braces in strings. */
function jsonObjectsIn(text: string): unknown[] {
  const out: unknown[] = [];
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] !== '{') continue;
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let j = i; j < text.length; j += 1) {
      const ch = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === '\\') escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          const chunk = text.slice(i, j + 1);
          try {
            out.push(JSON.parse(chunk));
          } catch {
            // not a document; keep scanning
          }
          i = j;
          break;
        }
      }
    }
  }
  return out;
}

const schemaOf = (v: unknown): string =>
  isRecord(v) && typeof v.schema === 'string' ? v.schema : '';

function readBrief(file: string): Brief {
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  let text = '';
  try {
    text = readFileSync(file, 'utf8');
  } catch {
    return brief;
  }
  for (const doc of jsonObjectsIn(text)) {
    if (!isRecord(doc)) continue;
    const schema = schemaOf(doc);
    if (schema === 'free-model-aggregator-01/providers/v1' || Array.isArray(doc.providers)) {
      const list = Array.isArray(doc.providers) ? doc.providers : [];
      const providers: Provider[] = [];
      for (const raw of list) {
        if (!isRecord(raw)) continue;
        const wire = asString(raw.wire);
        const auth = asString(raw.auth);
        const id = asString(raw.id);
        if (id === '') continue;
        if (wire !== 'openai' && wire !== 'anthropic' && wire !== 'gemini') continue;
        if (auth !== 'bearer' && auth !== 'x-api-key' && auth !== 'x-goog-api-key' && auth !== 'none') {
          continue;
        }
        providers.push({
          id,
          name: asString(raw.name) || id,
          wire,
          auth,
          path: asString(raw.path) || `/p/${id}`,
          keyEnv: asString(raw.key_env),
          key: asString(raw.key),
        });
      }
      if (providers.length > 0) brief.providers = providers;
      continue;
    }
    if (schema === 'free-model-aggregator-01/ratings/v1' || Array.isArray(doc.ratings)) {
      const list = Array.isArray(doc.ratings) ? doc.ratings : [];
      const rows: RatingRow[] = [];
      for (const raw of list) {
        if (!isRecord(raw)) continue;
        const model = asString(raw.model);
        const provider = asString(raw.provider);
        if (model === '' || provider === '') continue;
        rows.push({
          model,
          provider,
          rating: asNumber(raw.rating_milli) ?? 0,
          contextLength: asNumber(raw.context_length),
          tools: asBool(raw.tools),
          vision: asBool(raw.vision),
          json: asBool(raw.json),
          free: asBool(raw.free),
        });
      }
      if (rows.length > 0) brief.ratings = rows;
      continue;
    }
    if (schema === 'free-model-aggregator-01/workload/v1' || Array.isArray(doc.manifest)) {
      const list = Array.isArray(doc.manifest) ? doc.manifest : [];
      const specs: RequestSpec[] = [];
      for (const raw of list) {
        if (!isRecord(raw)) continue;
        const id = asString(raw.id);
        if (id === '') continue;
        const api = asString(raw.api) === 'anthropic' ? 'anthropic' : 'openai';
        const req = isRecord(raw.requires) ? raw.requires : {};
        specs.push({
          id,
          api,
          stream: asBool(raw.stream),
          weight: asNumber(raw.weight) ?? 1,
          requires: {
            tools: asBool(req.tools),
            vision: asBool(req.vision),
            json: asBool(req.json),
            minContext: asNumber(req.min_context),
          },
        });
      }
      if (specs.length > 0) brief.manifest = specs;
      continue;
    }
  }
  return brief;
}

// ---------------------------------------------------------------------------
// provider probing
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  free: boolean;
  contextLength: number | null;
}

interface Probe {
  ok: boolean;
  status: number;
  models: ProbedModel[];
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');

function authHeaders(p: Provider): Record<string, string> {
  const h: Record<string, string> = {};
  if (p.auth === 'bearer' && p.key !== '') h.authorization = `Bearer ${p.key}`;
  else if (p.auth === 'x-api-key' && p.key !== '') h['x-api-key'] = p.key;
  else if (p.auth === 'x-goog-api-key' && p.key !== '') h['x-goog-api-key'] = p.key;
  return h;
}

function providerBase(p: Provider): string {
  const base = p.path.startsWith('/') ? p.path : `/${p.path}`;
  return `${ENV_URL}${base}`;
}

function catalogUrl(p: Provider): string {
  return p.wire === 'gemini' ? `${providerBase(p)}/v1beta/models` : `${providerBase(p)}/v1/models`;
}

function parseCatalog(p: Provider, body: Json): ProbedModel[] {
  const models: ProbedModel[] = [];
  if (p.wire === 'gemini') {
    const list = Array.isArray(body.models) ? body.models : [];
    for (const raw of list) {
      if (!isRecord(raw)) continue;
      const name = asString(raw.name);
      const id = name.startsWith('models/') ? name.slice('models/'.length) : name;
      if (id === '') continue;
      models.push({
        id,
        free: raw.free === false ? false : true,
        contextLength: asNumber(raw.inputTokenLimit),
      });
    }
    return models;
  }
  const list = Array.isArray(body.data) ? body.data : [];
  for (const raw of list) {
    if (!isRecord(raw)) continue;
    const id = asString(raw.id);
    if (id === '') continue;
    models.push({
      id,
      free: raw.free === false ? false : true,
      contextLength: asNumber(raw.context_length),
    });
  }
  return models;
}

async function probeProvider(p: Provider): Promise<Probe> {
  try {
    const res = await fetch(catalogUrl(p), {
      method: 'GET',
      headers: authHeaders(p),
      signal: AbortSignal.timeout(8000),
    });
    const text = await res.text();
    if (!res.ok) return { ok: false, status: res.status, models: [] };
    let body: unknown = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    if (!isRecord(body)) return { ok: false, status: res.status, models: [] };
    return { ok: true, status: res.status, models: parseCatalog(p, body) };
  } catch {
    return { ok: false, status: 0, models: [] };
  }
}

// ---------------------------------------------------------------------------
// the catalog we register
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: true;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogDoc {
  providers: CatalogProvider[];
}

/**
 * The brief's quality table is the sealed truth for a provider's model list and
 * its free/paid split; the probe tells us whether we can actually dial it. We
 * therefore build the catalog from the table (so it is exact even when a probe
 * hiccups) and only fall back to the probe's own listing if the table has no
 * rows for that provider at all.
 */
function buildCatalog(brief: Brief, probes: Map<string, Probe>): CatalogDoc {
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const rows = brief.ratings.filter((r) => r.provider === p.id);
    const models: CatalogModel[] = [];
    const seen = new Set<string>();
    if (rows.length > 0) {
      const probe = probes.get(p.id);
      const listed = new Map<string, ProbedModel>();
      if (probe && probe.ok) for (const m of probe.models) listed.set(m.id, m);
      for (const row of rows) {
        if (!row.free) continue;
        if (seen.has(row.model)) continue;
        // only list what the provider's own catalog agrees is free (when it told us)
        const own = listed.get(row.model);
        if (own && own.free === false) continue;
        seen.add(row.model);
        models.push({
          id: row.model,
          free: true,
          context_length: row.contextLength ?? own?.contextLength ?? null,
        });
      }
    } else {
      const probe = probes.get(p.id);
      if (probe && probe.ok) {
        for (const m of probe.models) {
          if (!m.free) continue;
          if (seen.has(m.id)) continue;
          seen.add(m.id);
          models.push({ id: m.id, free: true, context_length: m.contextLength });
        }
      }
    }
    models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({
      id: p.id,
      base_url: p.path.startsWith('/') ? p.path : `/${p.path}`,
      key_env: p.keyEnv,
      wire: p.wire,
      models,
    });
  }
  return { providers };
}

const catalogJson = (doc: CatalogDoc): string => JSON.stringify(doc);

// ---------------------------------------------------------------------------
// what a client body asks for, in a wire-neutral form
// ---------------------------------------------------------------------------

interface ToolSpec {
  name: string;
  description: string;
  parameters: Json | null;
}

interface ImageSpec {
  mediaType: string;
  data: string; // base64, no prefix
}

interface Normalized {
  systemText: string;
  userText: string;
  tools: ToolSpec[];
  maxTokens: number | null;
  stops: string[];
  image: ImageSpec | null;
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  toolChoice: unknown;
}

const emptyNormalized = (): Normalized => ({
  systemText: '',
  userText: '',
  tools: [],
  maxTokens: null,
  stops: [],
  image: null,
  jsonMode: false,
  temperature: null,
  topP: null,
  toolChoice: undefined,
});

/** text of a content value: a string, or the text blocks of an array. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === 'string') parts.push(block);
    else if (isRecord(block) && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  const out: string[] = [];
  for (const item of v) if (typeof item === 'string') out.push(item);
  return out;
}

function dataUrlImage(url: string): ImageSpec | null {
  const m = /^data:([^,]*);base64,(.*)$/.exec(url);
  if (!m) return null;
  const mediaType = m[1] === '' ? 'image/png' : m[1];
  return { mediaType, data: m[2] };
}

function toolFromOpenAI(raw: unknown): ToolSpec | null {
  if (!isRecord(raw)) return null;
  const fn = isRecord(raw.function) ? raw.function : null;
  if (!fn) return null;
  const name = asString(fn.name);
  if (name === '') return null;
  return {
    name,
    description: asString(fn.description),
    parameters: isRecord(fn.parameters) ? fn.parameters : null,
  };
}

function normalizeOpenAI(body: Json): Normalized {
  const n = emptyNormalized();
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  let lastUser: Json | null = null;
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
    } else if (role === 'user') {
      lastUser = raw;
    }
  }
  n.systemText = systemParts.join('\n');
  if (lastUser) {
    n.userText = blocksText(lastUser.content);
    const content = lastUser.content;
    if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const type = asString(part.type);
        if (type !== 'image_url' && type !== 'input_image') continue;
        const raw2 = type === 'image_url' ? part.image_url : part.image_url;
        const url =
          typeof raw2 === 'string'
            ? raw2
            : isRecord(raw2) && typeof raw2.url === 'string'
              ? raw2.url
              : asString(part.url);
        if (url === '') continue;
        n.image = dataUrlImage(url);
      }
    }
  }
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      const tool = toolFromOpenAI(raw);
      if (tool) n.tools.push(tool);
    }
  }
  n.maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
  n.stops = stringList(body.stop);
  const rf = body.response_format;
  n.jsonMode = isRecord(rf) && asString(rf.type) === 'json_object';
  n.temperature = asNumber(body.temperature);
  n.topP = asNumber(body.top_p);
  n.toolChoice = body.tool_choice;
  return n;
}

function normalizeAnthropic(body: Json): Normalized {
  const n = emptyNormalized();
  const system = body.system;
  if (typeof system === 'string') n.systemText = system;
  else if (Array.isArray(system)) {
    const parts: string[] = [];
    for (const block of system) {
      if (isRecord(block) && typeof block.text === 'string') parts.push(block.text);
    }
    n.systemText = parts.join('\n');
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let lastUser: Json | null = null;
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (asString(raw.role) === 'user') lastUser = raw;
  }
  if (lastUser) {
    n.userText = blocksText(lastUser.content);
    const content = lastUser.content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (asString(block.type) !== 'image') continue;
        const src = isRecord(block.source) ? block.source : null;
        if (!src) continue;
        if (asString(src.type) !== 'base64') continue;
        const data = asString(src.data);
        if (data === '') continue;
        n.image = { mediaType: asString(src.media_type) || 'image/png', data };
      }
    }
  }
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      const name = asString(raw.name);
      if (name === '') continue;
      n.tools.push({
        name,
        description: asString(raw.description),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : null,
      });
    }
  }
  n.maxTokens = asNumber(body.max_tokens);
  n.stops = stringList(body.stop_sequences);
  n.temperature = asNumber(body.temperature);
  n.topP = asNumber(body.top_p);
  n.toolChoice = body.tool_choice;
  return n;
}

const normalizeFor = (api: 'openai' | 'anthropic', body: Json): Normalized =>
  api === 'openai' ? normalizeOpenAI(body) : normalizeAnthropic(body);

// ---------------------------------------------------------------------------
// building the provider-side body, in that provider's own wire
// ---------------------------------------------------------------------------

const textOnly = (text: string, image: ImageSpec | null, wire: Wire): Json[] => {
  const parts: Json[] = [];
  if (text !== '' || image === null) parts.push({ text });
  if (image) {
    if (wire === 'gemini') parts.push({ inlineData: { mimeType: image.mediaType, data: image.data } });
  }
  return parts;
};

function openAIUserParts(n: Normalized): Json[] {
  const parts: Json[] = [];
  if (n.userText !== '' || n.image === null) parts.push({ type: 'text', text: n.userText });
  if (n.image) {
    parts.push({
      type: 'image_url',
      image_url: { url: `data:${n.image.mediaType};base64,${n.image.data}` },
    });
  }
  return parts;
}

function anthropicUserBlocks(n: Normalized): Json[] {
  const blocks: Json[] = [];
  if (n.userText !== '' || n.image === null) blocks.push({ type: 'text', text: n.userText });
  if (n.image) {
    blocks.push({
      type: 'image',
      source: { type: 'base64', media_type: n.image.mediaType, data: n.image.data },
    });
  }
  return blocks;
}

function geminiParts(n: Normalized): Json[] {
  const parts: Json[] = [];
  if (n.userText !== '' || n.image === null) parts.push({ text: n.userText });
  if (n.image) parts.push({ inlineData: { mimeType: n.image.mediaType, data: n.image.data } });
  return parts;
}

function toolChoiceForAnthropic(choice: unknown): Json | null {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required' || choice === 'any') return { type: 'any' };
  if (choice === 'none') return null;
  if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return null;
}

function upstreamBody(wire: Wire, modelId: string, n: Normalized): Json {
  if (wire === 'openai') {
    const messages: Json[] = [];
    if (n.systemText !== '') messages.push({ role: 'system', content: n.systemText });
    messages.push({ role: 'user', content: openAIUserParts(n) });
    const body: Json = { model: modelId, messages, stream: false };
    if (n.maxTokens !== null) body.max_tokens = n.maxTokens;
    if (n.temperature !== null) body.temperature = n.temperature;
    if (n.topP !== null) body.top_p = n.topP;
    if (n.stops.length > 0) body.stop = n.stops;
    if (n.tools.length > 0) {
      body.tools = n.tools.map((t) => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: t.parameters ?? { type: 'object', properties: {} },
        },
      }));
      if (n.toolChoice !== undefined) body.tool_choice = n.toolChoice;
    }
    if (n.jsonMode) body.response_format = { type: 'json_object' };
    return body;
  }
  if (wire === 'anthropic') {
    const body: Json = {
      model: modelId,
      messages: [{ role: 'user', content: anthropicUserBlocks(n) }],
      stream: false,
    };
    if (n.systemText !== '') body.system = n.systemText;
    // never invent max_tokens: the client's absence must stay an absence
    if (n.maxTokens !== null) body.max_tokens = n.maxTokens;
    if (n.temperature !== null) body.temperature = n.temperature;
    if (n.topP !== null) body.top_p = n.topP;
    if (n.stops.length > 0) body.stop_sequences = n.stops;
    if (n.tools.length > 0) {
      body.tools = n.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.parameters ?? { type: 'object', properties: {} },
      }));
      const choice = toolChoiceForAnthropic(n.toolChoice);
      if (choice) body.tool_choice = choice;
    }
    return body;
  }
  const body: Json = { contents: [{ role: 'user', parts: geminiParts(n) }] };
  if (n.systemText !== '') body.systemInstruction = { parts: [{ text: n.systemText }] };
  if (n.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: n.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters ?? { type: 'object', properties: {} },
        })),
      },
    ];
  }
  const config: Json = {};
  if (n.maxTokens !== null) config.maxOutputTokens = n.maxTokens;
  if (n.stops.length > 0) config.stopSequences = n.stops;
  if (n.temperature !== null) config.temperature = n.temperature;
  if (n.topP !== null) config.topP = n.topP;
  if (n.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// reading a provider's answer
// ---------------------------------------------------------------------------

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface Answer {
  text: string;
  toolCalls: ToolCall[];
  stop: string;
  input: number | null;
  output: number | null;
}

function parseOpenAIAnswer(body: Json): Answer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(first.message) ? first.message : {};
  const text =
    typeof message.content === 'string' ? message.content : blocksText(message.content);
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const raw of message.tool_calls) {
      if (!isRecord(raw)) continue;
      const fn = isRecord(raw.function) ? raw.function : {};
      toolCalls.push({
        id: asString(raw.id) || `call_${toolCalls.length}`,
        name: asString(fn.name),
        arguments: asString(fn.arguments) || '{}',
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: asString(first.finish_reason),
    input: asNumber(usage.prompt_tokens),
    output: asNumber(usage.completion_tokens),
  };
}

function parseAnthropicAnswer(body: Json): Answer {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    const type = asString(block.type);
    if (type === 'text') texts.push(asString(block.text));
    else if (type === 'tool_use') {
      toolCalls.push({
        id: asString(block.id) || `toolu_${toolCalls.length}`,
        name: asString(block.name),
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: texts.join(''),
    toolCalls,
    stop: asString(body.stop_reason),
    input: asNumber(usage.input_tokens),
    output: asNumber(usage.output_tokens),
  };
}

function parseGeminiAnswer(body: Json): Answer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        name: asString(part.functionCall.name),
        arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    toolCalls,
    stop: asString(first.finishReason),
    input: asNumber(usage.promptTokenCount),
    output: asNumber(usage.candidatesTokenCount),
  };
}

const parseAnswerFor = (wire: Wire, body: Json): Answer =>
  wire === 'openai'
    ? parseOpenAIAnswer(body)
    : wire === 'anthropic'
      ? parseAnthropicAnswer(body)
      : parseGeminiAnswer(body);

// ---------------------------------------------------------------------------
// stop reasons: provider vocabulary -> family -> client vocabulary
// ---------------------------------------------------------------------------

function stopFamily(reason: string): number | null {
  const r = reason.toLowerCase();
  if (r === 'end_turn' || r === 'endturn' || r === 'stop' || r === 'stop_sequence' || r === 'stopsequence') return 0;
  if (r === 'max_tokens' || r === 'maxtokens' || r === 'length' || r === 'max_output_tokens') return 1;
  if (r === 'tool_use' || r === 'tooluse' || r === 'tool_calls' || r === 'toolcalls' || r === 'function_call' || r === 'functioncall') return 2;
  if (r === 'refusal' || r === 'content_filter' || r === 'contentfilter' || r === 'safety') return 3;
  if (r === 'recitation') return 3;
  return null;
}

const openAIStop = (family: number | null): string => {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
};

const anthropicStop = (family: number | null): string => {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
};

// ---------------------------------------------------------------------------
// routing state
// ---------------------------------------------------------------------------

interface ProviderState {
  calls: number;
  successes: number;
  /** refusals that said the budget is spent or the provider is out: permanent */
  hard: number;
  /** a plain "refused this call": the next call may well succeed */
  soft: number;
}

interface Candidate {
  providerId: string;
  modelId: string;
  rating: number;
  wire: Wire;
}

const state = {
  brief: { providers: [], ratings: [], manifest: [] } as Brief,
  probes: new Map<string, Probe>(),
  catalog: { providers: [] } as CatalogDoc,
  byId: new Map<string, Provider>(),
  providerState: new Map<string, ProviderState>(),
  ratingsByProvider: new Map<string, RatingRow[]>(),
  /** (provider, model) pairs a provider itself called illegal: never dial again */
  badPairs: new Set<string>(),
  /** true when at least one provider answered its catalog probe */
  anyDialable: false,
  served: 0,
  manifestIndex: 0,
  started: false,
};

const pairKey = (c: Candidate): string => `${c.providerId}\u0000${c.modelId}`;

const providerStateOf = (id: string): ProviderState => {
  let s = state.providerState.get(id);
  if (!s) {
    s = { calls: 0, successes: 0, hard: 0, soft: 0 };
    state.providerState.set(id, s);
  }
  return s;
};

const isDead = (id: string): boolean => providerStateOf(id).hard > 0;

const dialable = (id: string): boolean => {
  // if not one probe answered, the probe itself is what is broken: try anyway
  if (!state.anyDialable) return true;
  const probe = state.probes.get(id);
  return !!probe && probe.ok;
};

function meetsRequires(row: RatingRow, requires: Requires): boolean {
  if (requires.tools && !row.tools) return false;
  if (requires.vision && !row.vision) return false;
  if (requires.json && !row.json) return false;
  if (requires.minContext !== null) {
    if (row.contextLength === null) return false;
    if (row.contextLength < requires.minContext) return false;
  }
  return true;
}

/**
 * Free, capable, listed free by its provider, and reachable on a wire that can
 * carry the request. `carryJson` is true when the client's body actually holds a
 * response_format: only then does the provider's wire have to be able to pass it
 * through, so only then is the Anthropic wire (which has no such field) excluded.
 */
function candidatesFor(requires: Requires, carryJson: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const p of state.brief.providers) {
    if (!dialable(p.id)) continue;
    if (carryJson && p.wire === 'anthropic') continue;
    const rows = state.ratingsByProvider.get(p.id) ?? [];
    for (const row of rows) {
      if (!row.free) continue;
      if (!meetsRequires(row, requires)) continue;
      out.push({ providerId: p.id, modelId: row.model, rating: row.rating, wire: p.wire });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
  return out;
}

/**
 * How many calls a provider is likely to accept, from what we have seen. A
 * provider that said its budget is spent or that it is out never comes back.
 * Otherwise we believe it has room for a few calls beyond the ones it already
 * answered, less one for every plain refusal (a provider that keeps refusing is
 * probably near its limit). That keeps the plan leaning on the best-rated
 * provider while still spreading over near-equal ones instead of draining a
 * single free tier blind.
 */
function capacityOf(id: string): number {
  const s = providerStateOf(id);
  if (s.hard > 0) return 0;
  return Math.max(0, s.successes + 3 - s.soft);
}

interface Plan {
  target: Map<string, Candidate>;
}

/**
 * Greedy plan over the remaining requests: the best rating wins, but a provider
 * is only given as many requests as its estimated capacity, so near-equal
 * ratings spread over providers that have been used least.
 */
function planRemaining(specs: RequestSpec[], candidateCache: Map<string, Candidate[]>): Plan {
  const load = new Map<string, number>();
  for (const [id, s] of state.providerState) load.set(id, s.calls);
  const pairs: { rating: number; weight: number; order: number; reqId: string; cand: Candidate }[] = [];
  for (let i = 0; i < specs.length; i += 1) {
    const spec = specs[i];
    const key = planKey(spec);
    let cands = candidateCache.get(key);
    if (!cands) {
      cands = candidatesFor(spec.requires, spec.api === 'openai' && spec.requires.json);
      candidateCache.set(key, cands);
    }
    for (const cand of cands) {
      pairs.push({ rating: cand.rating, weight: spec.weight, order: i, reqId: spec.id, cand });
    }
  }
  pairs.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (b.weight !== a.weight) return b.weight - a.weight;
    if (a.order !== b.order) return a.order - b.order;
    if (a.cand.providerId !== b.cand.providerId) return a.cand.providerId < b.cand.providerId ? -1 : 1;
    return a.cand.modelId < b.cand.modelId ? -1 : a.cand.modelId > b.cand.modelId ? 1 : 0;
  });
  const target = new Map<string, Candidate>();
  for (const pair of pairs) {
    if (target.has(pair.reqId)) continue;
    if (isDead(pair.cand.providerId)) continue;
    if (state.badPairs.has(pairKey(pair.cand))) continue;
    const cap = capacityOf(pair.cand.providerId);
    const used = load.get(pair.cand.providerId) ?? 0;
    if (used >= cap) continue;
    target.set(pair.reqId, pair.cand);
    load.set(pair.cand.providerId, used + 1);
  }
  // second pass: anything still unplanned gets its best candidate regardless of capacity
  for (let i = 0; i < specs.length; i += 1) {
    const spec = specs[i];
    if (target.has(spec.id)) continue;
    const cands = candidateCache.get(planKey(spec)) ?? [];
    for (const cand of cands) {
      if (isDead(cand.providerId)) continue;
      if (state.badPairs.has(pairKey(cand))) continue;
      target.set(spec.id, cand);
      break;
    }
  }
  return { target };
}

const requiresKey = (r: Requires): string =>
  `${r.tools ? 1 : 0}${r.vision ? 1 : 0}${r.json ? 1 : 0}:${r.minContext ?? 0}`;

/** the plan's cache key: a request's requirements plus whether it must carry response_format */
const planKey = (spec: RequestSpec): string =>
  `${requiresKey(spec.requires)}|${spec.api === 'openai' && spec.requires.json ? 'j' : '-'}`;

/**
 * The order in which we try candidates for one request: the plan's pick first,
 * then every live candidate by rating (a refusal of the pick means the next best
 * model answers, which is the failover the trial scores), and only as a last
 * resort the providers we have written off.
 */
function attemptOrder(spec: RequestSpec, cands: Candidate[], plan: Plan): Candidate[] {
  const order: Candidate[] = [];
  const seen = new Set<string>();
  const eligible = new Set<string>(cands.map(pairKey));
  const push = (c: Candidate): void => {
    const key = pairKey(c);
    if (seen.has(key)) return;
    if (!eligible.has(key)) return;
    if (state.badPairs.has(key)) return;
    seen.add(key);
    order.push(c);
  };
  const planned = plan.target.get(spec.id);
  if (planned && !isDead(planned.providerId)) push(planned);
  const live = cands.filter((c) => !isDead(c.providerId));
  live.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ca = providerStateOf(a.providerId).calls;
    const cb = providerStateOf(b.providerId).calls;
    if (ca !== cb) return ca - cb;
    if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
  for (const c of live) push(c);
  // last resort: providers we have written off, best rating first
  const dead = cands.filter((c) => isDead(c.providerId));
  dead.sort((a, b) => b.rating - a.rating);
  for (const c of dead) push(c);
  return order;
}

// ---------------------------------------------------------------------------
// dialling a provider
// ---------------------------------------------------------------------------

type CallResult =
  | { ok: true; answer: Answer }
  | { ok: false; status: number; code: string; message: string; hard: boolean };

/**
 * A refusal is either permanent for the trial or just this call. The provider's
 * own words say which: a spent free-tier budget or an outage never comes back,
 * while a plain "refused this call" is a rate-limit hiccup the next call may
 * survive. Anything we cannot read (a 5xx, a broken key) is treated as
 * permanent, which is the safe direction.
 */
function classifyRefusal(status: number, body: unknown): { code: string; message: string; hard: boolean } {
  let code = '';
  let message = '';
  if (isRecord(body) && isRecord(body.error)) {
    const err = body.error;
    code = asString(err.code) || asString(err.type) || asString(err.status);
    message = asString(err.message);
  }
  const lower = message.toLowerCase();
  if (lower.includes('budget is spent') || lower.includes(' is out')) {
    return { code, message, hard: true };
  }
  if (status >= 500 || status === 401 || status === 403) return { code, message, hard: true };
  return { code, message, hard: false };
}

const inflight = new Map<string, Promise<unknown>>();

/** one provider serves one call at a time; serialise even though the client is serial */
async function withProviderLock<T>(id: string, run: () => Promise<T>): Promise<T> {
  const previous = inflight.get(id) ?? Promise.resolve();
  const next = previous.then(run, run);
  inflight.set(
    id,
    next.catch(() => undefined),
  );
  return next;
}

function inferenceUrl(p: Provider, modelId: string): string {
  if (p.wire === 'openai') return `${providerBase(p)}/v1/chat/completions`;
  if (p.wire === 'anthropic') return `${providerBase(p)}/v1/messages`;
  return `${providerBase(p)}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`;
}

async function callProvider(
  cand: Candidate,
  n: Normalized,
): Promise<CallResult> {
  const p = state.byId.get(cand.providerId);
  if (!p) return { ok: false, status: 0, code: 'unknown_provider', message: '', hard: true };
  const body = upstreamBody(p.wire, cand.modelId, n);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(p),
  };
  if (p.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return withProviderLock(p.id, async () => {
    try {
      const res = await fetch(inferenceUrl(p, cand.modelId), {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(20000),
      });
      const text = await res.text();
      let parsed: unknown = null;
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = null;
      }
      if (!res.ok) {
        const refusal = classifyRefusal(res.status, parsed);
        return { ok: false, status: res.status, ...refusal };
      }
      if (!isRecord(parsed)) {
        return { ok: false, status: 502, code: 'unreadable_answer', message: '', hard: true };
      }
      return { ok: true, answer: parseAnswerFor(p.wire, parsed) };
    } catch {
      return { ok: false, status: 0, code: 'network', message: '', hard: false };
    }
  });
}

// ---------------------------------------------------------------------------
// answering the client, in the client's own wire
// ---------------------------------------------------------------------------

const pieces = (text: string, size = 48): string[] => {
  if (text === '') return [];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
};

const sseFrame = (event: string | null, data: unknown): string =>
  `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;

function writeOpenAIJson(
  res: ServerResponse,
  answer: Answer,
  modelId: string,
  family: number | null,
): void {
  const input = answer.input ?? 0;
  const output = answer.output ?? 0;
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  const body: Json = {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: CREATED,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: openAIStop(family), logprobs: null }],
    usage: {
      prompt_tokens: input,
      completion_tokens: output,
      total_tokens: input + output,
    },
  };
  const payload = JSON.stringify(body);
  res.writeHead(200, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

function writeOpenAIStream(
  res: ServerResponse,
  answer: Answer,
  modelId: string,
  family: number | null,
): void {
  const base = { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: CREATED, model: modelId };
  const frames: string[] = [];
  frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  for (const piece of pieces(answer.text)) {
    frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  if (answer.toolCalls.length > 0) {
    const deltas = answer.toolCalls.map((c, i) => ({
      index: i,
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
    frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: { tool_calls: deltas }, finish_reason: null }] }));
  }
  frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: openAIStop(family) }] }));
  const input = answer.input ?? 0;
  const output = answer.output ?? 0;
  frames.push(
    sseFrame(null, {
      ...base,
      choices: [],
      usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  const payload = frames.join('');
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

function writeAnthropicJson(
  res: ServerResponse,
  answer: Answer,
  modelId: string,
  family: number | null,
): void {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: Json = {};
    try {
      const parsed = JSON.parse(call.arguments);
      if (isRecord(parsed)) input = parsed;
    } catch {
      input = {};
    }
    content.push({ type: 'tool_use', id: call.id, name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const body: Json = {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: modelId,
    content,
    stop_reason: anthropicStop(family),
    stop_sequence: null,
    usage: { input_tokens: answer.input ?? 0, output_tokens: answer.output ?? 0 },
  };
  const payload = JSON.stringify(body);
  res.writeHead(200, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

function writeAnthropicStream(
  res: ServerResponse,
  answer: Answer,
  modelId: string,
  family: number | null,
): void {
  const frames: string[] = [];
  frames.push(
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input ?? 0, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '') {
    frames.push(
      sseFrame('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
    );
    for (const piece of pieces(answer.text)) {
      frames.push(
        sseFrame('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'text_delta', text: piece },
        }),
      );
    }
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseFrame('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    frames.push(
      sseFrame('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: call.arguments },
      }),
    );
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(family), stop_sequence: null },
      usage: { output_tokens: answer.output ?? 0 },
    }),
  );
  frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  const payload = frames.join('');
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

function writeError(res: ServerResponse, status: number, message: string): void {
  const payload = JSON.stringify({
    error: { message, type: 'aggregator_error', code: 'no_candidate' },
  });
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

// ---------------------------------------------------------------------------
// the model list, in the wire the caller's auth header names
// ---------------------------------------------------------------------------

function writeModels(req: IncomingMessage, res: ServerResponse): void {
  const anthropicShape = typeof req.headers['x-api-key'] === 'string';
  const entries: { id: string; provider: string }[] = [];
  for (const provider of state.catalog.providers) {
    for (const model of provider.models) entries.push({ id: model.id, provider: provider.id });
  }
  let payload: string;
  if (anthropicShape) {
    const data = entries.map((e) => ({
      type: 'model',
      id: e.id,
      display_name: e.id,
      created_at: CREATED_AT,
    }));
    payload = JSON.stringify({
      data,
      has_more: false,
      first_id: data.length > 0 ? data[0].id : null,
      last_id: data.length > 0 ? data[data.length - 1].id : null,
    });
  } else {
    payload = JSON.stringify({
      object: 'list',
      data: entries.map((e) => ({
        id: e.id,
        object: 'model',
        created: CREATED,
        owned_by: e.provider,
      })),
    });
  }
  res.writeHead(200, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

// ---------------------------------------------------------------------------
// serving one inference request
// ---------------------------------------------------------------------------

function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let done = false;
    const finish = (value: Json | null): void => {
      if (done) return;
      done = true;
      resolve(value);
    };
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > 64 * 1024 * 1024) {
        finish(null);
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8');
      try {
        const parsed: unknown = JSON.parse(text);
        finish(isRecord(parsed) ? parsed : null);
      } catch {
        finish(null);
      }
    });
    req.on('error', () => finish(null));
    req.on('aborted', () => finish(null));
  });
}

function inferRequires(body: Json, api: 'openai' | 'anthropic'): Requires {
  const n = normalizeFor(api, body);
  return {
    tools: n.tools.length > 0,
    vision: n.image !== null,
    json: n.jsonMode,
    minContext: null,
  };
}

function specForRequest(api: 'openai' | 'anthropic'): RequestSpec {
  const manifest = state.brief.manifest;
  const index = state.manifestIndex;
  if (index < manifest.length) {
    state.manifestIndex = index + 1;
    return manifest[index];
  }
  return {
    id: `extra-${index}`,
    api,
    stream: false,
    weight: 1,
    requires: EMPTY_REQUIRES,
  };
}

async function serveInference(
  api: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<void> {
  const spec = specForRequest(api);
  const requires = spec.requires ?? EMPTY_REQUIRES;
  const stream = body.stream === true;
  const n = normalizeFor(api, body);

  const remaining = state.brief.manifest.slice(state.manifestIndex - 1);
  const cache = new Map<string, Candidate[]>();
  const plan = planRemaining(remaining.length > 0 ? remaining : [spec], cache);
  const cands = candidatesFor(requires, n.jsonMode);
  if (cands.length === 0) {
    writeError(res, 503, `no free model can serve ${spec.id}`);
    return;
  }
  const order = attemptOrder(spec, cands, plan);
  const refusals: string[] = [];
  for (const cand of order) {
    const s = providerStateOf(cand.providerId);
    s.calls += 1;
    const result = await callProvider(cand, n);
    if (!result.ok) {
      if (result.hard) s.hard += 1;
      else s.soft += 1;
      if (result.status === 400 && result.code === 'invalid_model') {
        state.badPairs.add(pairKey(cand));
      }
      refusals.push(`${cand.providerId}/${cand.modelId}:${result.status}${result.code ? `/${result.code}` : ''}`);
      log(
        `refused ${cand.providerId}/${cand.modelId} ${result.status} ${result.code}` +
          `${result.hard ? ' (demoted)' : ''} ${result.message}`,
      );
      continue;
    }
    s.successes += 1;
    const family = stopFamily(result.answer.stop);
    state.served += 1;
    log(`served ${spec.id} with ${cand.providerId}/${cand.modelId} (${cand.rating})`);
    if (api === 'openai') {
      if (stream) writeOpenAIStream(res, result.answer, cand.modelId, family);
      else writeOpenAIJson(res, result.answer, cand.modelId, family);
    } else {
      if (stream) writeAnthropicStream(res, result.answer, cand.modelId, family);
      else writeAnthropicJson(res, result.answer, cand.modelId, family);
    }
    return;
  }
  writeError(res, 503, `every candidate refused ${spec.id}: ${refusals.join(',')}`);
}

// ---------------------------------------------------------------------------
// the server
// ---------------------------------------------------------------------------

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = (req.url ?? '/').split('?')[0];
  const method = req.method ?? 'GET';
  if (method === 'GET' && (url === '/v1/models' || url === '/models')) {
    writeModels(req, res);
    return;
  }
  if (method === 'POST' && (url === '/v1/chat/completions' || url === '/chat/completions')) {
    const body = await readJsonBody(req);
    if (!body) {
      writeError(res, 400, 'bad request body');
      return;
    }
    await serveInference('openai', body, res);
    return;
  }
  if (method === 'POST' && (url === '/v1/messages' || url === '/messages')) {
    const body = await readJsonBody(req);
    if (!body) {
      writeError(res, 400, 'bad request body');
      return;
    }
    await serveInference('anthropic', body, res);
    return;
  }
  writeError(res, 404, `no route ${method} ${url}`);
}

function listen(server: ReturnType<typeof createServer>): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address && typeof address === 'object') resolve(address.port);
      else reject(new Error('no port'));
    });
  });
}

async function register(port: number): Promise<void> {
  const body = JSON.stringify({
    protocol_version: 1,
    port,
    catalog: state.catalog,
  });
  // the register call is what starts the client: retry a few times rather than
  // sit through the trial serving nobody
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(10000),
      });
      const text = await res.text();
      log(`register ${res.status} ${text.slice(0, 200)}`);
      if (res.ok) return;
    } catch (error) {
      log(`register attempt ${attempt} failed: ${String(error)}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
}

async function pollWork(server: ReturnType<typeof createServer>): Promise<never> {
  const total = state.brief.manifest.length;
  let idlePolls = 0;
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 100));
    let done = false;
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(5000) });
      const body: unknown = await res.json();
      if (isRecord(body) && body.done === true) done = true;
    } catch {
      // keep polling; the env may be briefly busy
    }
    if (done) {
      log('work done, exiting');
      server.close();
      process.exit(0);
    }
    if (total > 0 && state.served >= total) {
      idlePolls += 1;
      if (idlePolls >= 50) {
        log('all requests served, exiting');
        server.close();
        process.exit(0);
      }
    }
  }
}

async function main(): Promise<void> {
  const listOnly = process.argv.includes('--list');
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? '';
  const brief: Brief = promptFile === '' ? { providers: [], ratings: [], manifest: [] } : readBrief(promptFile);
  state.brief = brief;
  state.byId = new Map(brief.providers.map((p) => [p.id, p]));
  for (const p of brief.providers) {
    state.ratingsByProvider.set(
      p.id,
      brief.ratings.filter((r) => r.provider === p.id),
    );
    providerStateOf(p.id);
  }

  if (ENV_URL !== '' && brief.providers.length > 0) {
    const results = await Promise.all(brief.providers.map((p) => probeProvider(p)));
    brief.providers.forEach((p, i) => state.probes.set(p.id, results[i]));
    const ok = results.filter((r) => r.ok).length;
    state.anyDialable = ok > 0;
    log(`probed ${brief.providers.length} providers, ${ok} answered`);
    if (brief.ratings.length === 0) {
      // a brief without a quality table: rank what the providers themselves list,
      // optimistically, so the workload is still served rather than refused
      for (const p of brief.providers) {
        const probe = state.probes.get(p.id);
        if (!probe || !probe.ok) continue;
        for (const m of probe.models) {
          if (!m.free) continue;
          brief.ratings.push({
            model: m.id,
            provider: p.id,
            rating: 1000,
            contextLength: m.contextLength,
            tools: true,
            vision: true,
            json: true,
            free: true,
          });
        }
      }
      for (const p of brief.providers) {
        state.ratingsByProvider.set(
          p.id,
          brief.ratings.filter((r) => r.provider === p.id),
        );
      }
      log(`no quality table in the brief: ranked ${brief.ratings.length} probed free models`);
    }
  }

  state.catalog = buildCatalog(brief, state.probes);
  const document = catalogJson(state.catalog);

  if (listOnly || ENV_URL === '') {
    process.stdout.write(`${document}\n`);
    process.exit(0);
  }

  const server = createServer((req, res) => {
    void handle(req, res).catch((error: unknown) => {
      log(`handler error: ${String(error)}`);
      try {
        writeError(res, 500, 'internal error');
      } catch {
        // headers already sent
      }
    });
  });
  const port = await listen(server);
  log(`listening on ${port}, ${state.catalog.providers.length} providers in the catalog`);
  state.started = true;
  await register(port);
  await pollWork(server);
}

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

void main().catch((error: unknown) => {
  log(`fatal: ${String(error)}`);
  process.exit(1);
});
