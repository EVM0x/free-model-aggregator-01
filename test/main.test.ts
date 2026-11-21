/**
 * Tests for the aggregator's pure surfaces.
 *
 * The module's docstring makes four promises — standard library only, no wall
 * clock, no randomness, "the same brief and the same provider answers always
 * produce the same state export" — and describes a specific routing contract.
 * Nothing checked any of it, and the file could not even be imported: `main()`
 * ran at module scope and called `process.exit`. It now runs only when invoked
 * directly, so the pure functions below can be exercised.
 *
 * Run: node --test test/*.test.ts
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  jsonObjectsIn,
  readBrief,
  blocksText,
  stringList,
  dataUrlImage,
  normalizeFor,
  upstreamBody,
  parseAnswerFor,
  stopFamily,
  openAIStop,
  anthropicStop,
  meetsRequires,
  candidatesFor,
  capacityOf,
  planRemaining,
  attemptOrder,
  classifyRefusal,
  inferRequires,
  providerStateOf,
  state,
  pairKey,
  openAIUserParts,
  anthropicUserBlocks,
  geminiParts,
  toolChoiceForAnthropic,
} from '../main.ts';

type Json = Record<string, unknown>;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

const resetState = (): void => {
  state.brief = { providers: [], ratings: [], manifest: [] };
  state.byId = new Map();
  state.probes = new Map();
  state.providerState = new Map();
  state.ratingsByProvider = new Map();
  state.badPairs = new Set();
  state.anyDialable = false;
  state.served = 0;
  state.manifestIndex = 0;
  state.started = false;
};

const provider = (id: string, wire: string, extra: Json = {}): Json => ({
  id,
  name: id,
  wire,
  auth: 'bearer',
  path: `/p/${id}`,
  keyEnv: '',
  key: '',
  ...extra,
});

const rating = (model: string, prov: string, r: number, extra: Json = {}): Json => ({
  model,
  provider: prov,
  rating: r,
  contextLength: 128000,
  tools: true,
  vision: true,
  json: true,
  free: true,
  ...extra,
});

/** Load a brief straight into the module's routing state. */
const load = (brief: { providers: Json[]; ratings: Json[]; manifest?: Json[] }): void => {
  resetState();
  const providers = brief.providers as never[];
  const ratings = brief.ratings as never[];
  state.brief = {
    providers,
    ratings,
    manifest: (brief.manifest ?? []) as never[],
  };
  state.byId = new Map(providers.map((p: Json) => [p.id as string, p as never]));
  for (const p of providers as Json[]) {
    state.ratingsByProvider.set(
      p.id as string,
      (ratings as Json[]).filter((r) => r.provider === p.id) as never[],
    );
    providerStateOf(p.id as string);
  }
};

const spec = (id: string, extra: Json = {}): Json => ({
  id,
  api: 'openai',
  stream: false,
  weight: 1,
  requires: { tools: false, vision: false, json: false, minContext: null },
  ...extra,
});

const tmp = (name: string, body: string): string => {
  const dir = mkdtempSync(join(tmpdir(), 'fma-'));
  const path = join(dir, name);
  writeFileSync(path, body);
  return path;
};

// ---------------------------------------------------------------------------
// jsonObjectsIn: three JSON documents inside prose
// ---------------------------------------------------------------------------

test('jsonObjectsIn finds every top-level object in order', () => {
  const text = 'a {"x":1} b {"y":2} c';
  assert.deepEqual(jsonObjectsIn(text), [{ x: 1 }, { y: 2 }]);
});

test('jsonObjectsIn keeps a nested object whole', () => {
  const text = '{"a":{"b":{"c":1}},"d":2}';
  assert.deepEqual(jsonObjectsIn(text), [{ a: { b: { c: 1 } }, d: 2 }]);
});

test('jsonObjectsIn does not end an object on a brace inside a string', () => {
  const text = '{"a":"} not the end {"}';
  assert.deepEqual(jsonObjectsIn(text), [{ a: '} not the end {' }]);
});

test('jsonObjectsIn survives an escaped quote before a brace', () => {
  const text = '{"a":"quote \\" then } brace"}';
  assert.deepEqual(jsonObjectsIn(text), [{ a: 'quote " then } brace' }]);
});

test('jsonObjectsIn skips a brace that does not parse and keeps scanning', () => {
  const text = '{oops} {"ok":true}';
  assert.deepEqual(jsonObjectsIn(text), [{ ok: true }]);
});

test('jsonObjectsIn returns nothing for text with no object', () => {
  assert.deepEqual(jsonObjectsIn('no json here'), []);
  assert.deepEqual(jsonObjectsIn(''), []);
});

test('jsonObjectsIn reads an array of objects as separate objects', () => {
  assert.deepEqual(jsonObjectsIn('[{"a":1},{"b":2}]'), [{ a: 1 }, { b: 2 }]);
});

// ---------------------------------------------------------------------------
// readBrief
// ---------------------------------------------------------------------------

test('readBrief returns an empty brief when the file is missing', () => {
  const brief = readBrief('/nonexistent/brief.txt');
  assert.deepEqual(brief, { providers: [], ratings: [], manifest: [] });
});

test('readBrief picks up the three documents by their shape', () => {
  const text = [
    'Providers:',
    '{"schema":"free-model-aggregator-01/providers/v1","providers":[' +
      JSON.stringify(provider('alpha', 'openai')) + ']}',
    'Quality:',
    '{"schema":"free-model-aggregator-01/ratings/v1","ratings":[' +
      JSON.stringify(rating('m1', 'alpha', 900)) + ']}',
    'Workload:',
    '{"schema":"free-model-aggregator-01/workload/v1","manifest":[' +
      JSON.stringify(spec('r1')) + ']}',
  ].join('\n');
  const brief = readBrief(tmp('brief.txt', text));
  assert.equal(brief.providers.length, 1);
  assert.equal(brief.providers[0].id, 'alpha');
  assert.equal(brief.ratings.length, 1);
  assert.equal(brief.ratings[0].model, 'm1');
  assert.equal(brief.manifest.length, 1);
  assert.equal(brief.manifest[0].id, 'r1');
});

test('readBrief defaults a provider name to its id and its path to /p/<id>', () => {
  const text = '{"providers":[{"id":"alpha","wire":"openai","auth":"bearer"}]}';
  const brief = readBrief(tmp('b.txt', text));
  assert.equal(brief.providers[0].name, 'alpha');
  assert.equal(brief.providers[0].path, '/p/alpha');
});

test('readBrief drops a provider with an unknown wire or auth', () => {
  const text =
    '{"providers":[' +
    '{"id":"a","wire":"carrier-pigeon","auth":"bearer"},' +
    '{"id":"b","wire":"openai","auth":"magic"},' +
    '{"id":"c","wire":"openai","auth":"bearer"}]}';
  const brief = readBrief(tmp('b.txt', text));
  assert.deepEqual(brief.providers.map((p) => p.id), ['c']);
});

test('readBrief reads a workload entry and its requirements', () => {
  const text =
    '{"manifest":[{"id":"r1","api":"anthropic","stream":true,"weight":3,' +
    '"requires":{"tools":true,"vision":false,"json":true,"min_context":32000}}]}';
  const got = readBrief(tmp('b.txt', text)).manifest;
  assert.equal(got.length, 1);
  assert.equal(got[0].api, 'anthropic');
  assert.equal(got[0].stream, true);
  assert.equal(got[0].weight, 3);
  assert.deepEqual(got[0].requires, { tools: true, vision: false, json: true, minContext: 32000 });
});

test('readBrief defaults an unknown api to openai and a missing weight to one', () => {
  const got = readBrief(tmp('b.txt', '{"manifest":[{"id":"r1","api":"grpc"}]}')).manifest;
  assert.equal(got[0].api, 'openai');
  assert.equal(got[0].weight, 1);
  assert.equal(got[0].stream, false);
  assert.equal(got[0].requires.minContext, null);
});

test('readBrief drops a workload entry with no id', () => {
  assert.deepEqual(readBrief(tmp('b.txt', '{"manifest":[{"api":"openai"}]}')).manifest, []);
});

test('readBrief drops a provider with no id', () => {
  const text = '{"providers":[{"wire":"openai","auth":"bearer"}]}';
  assert.deepEqual(readBrief(tmp('b.txt', text)).providers, []);
});

test('readBrief accepts all four auth schemes and all three wires', () => {
  const text =
    '{"providers":[' +
    ['openai', 'anthropic', 'gemini']
      .flatMap((wire, i) =>
        ['bearer', 'x-api-key', 'x-goog-api-key', 'none'].map(
          (auth, j) => `{"id":"p${i}${j}","wire":"${wire}","auth":"${auth}"}`,
        ),
      )
      .join(',') +
    ']}';
  assert.equal(readBrief(tmp('b.txt', text)).providers.length, 12);
});

// ---------------------------------------------------------------------------
// small content helpers
// ---------------------------------------------------------------------------

test('blocksText flattens a string, a block list and a mixed list', () => {
  assert.equal(blocksText('hello'), 'hello');
  assert.equal(blocksText([{ text: 'a' }, { text: 'b' }]), 'a\nb');
  assert.equal(blocksText(['a', { text: 'b' }]), 'a\nb');
  assert.equal(blocksText(undefined), '');
  assert.equal(blocksText(42), '');
});

test('stringList takes a bare string as a one-item list', () => {
  assert.deepEqual(stringList('stop'), ['stop']);
  assert.deepEqual(stringList(['a', 'b']), ['a', 'b']);
  assert.deepEqual(stringList(['a', 1, null]), ['a']);
  assert.deepEqual(stringList(null), []);
});

test('dataUrlImage reads a base64 data URL', () => {
  assert.deepEqual(dataUrlImage('data:image/jpeg;base64,QUJD'), {
    mediaType: 'image/jpeg',
    data: 'QUJD',
  });
});

test('dataUrlImage defaults an empty media type to image/png', () => {
  assert.deepEqual(dataUrlImage('data:;base64,QUJD'), { mediaType: 'image/png', data: 'QUJD' });
});

test('dataUrlImage refuses anything that is not a base64 data URL', () => {
  assert.equal(dataUrlImage('https://example.com/a.png'), null);
  assert.equal(dataUrlImage('data:image/png,notbase64'), null);
  assert.equal(dataUrlImage(''), null);
});

// ---------------------------------------------------------------------------
// normalize: the client wire into one internal shape
// ---------------------------------------------------------------------------

test('normalizeFor joins every system message and keeps the last user one', () => {
  const n = normalizeFor('openai', {
    messages: [
      { role: 'system', content: 'first' },
      { role: 'system', content: 'second' },
      { role: 'user', content: 'old question' },
      { role: 'assistant', content: 'answer' },
      { role: 'user', content: 'new question' },
    ],
  });
  assert.equal(n.systemText, 'first\nsecond');
  assert.equal(n.userText, 'new question');
});

test('normalizeFor reads max_tokens and falls back to max_completion_tokens', () => {
  assert.equal(normalizeFor('openai', { max_tokens: 100 }).maxTokens, 100);
  assert.equal(normalizeFor('openai', { max_completion_tokens: 200 }).maxTokens, 200);
  assert.equal(
    normalizeFor('openai', { max_tokens: 100, max_completion_tokens: 200 }).maxTokens,
    100,
  );
  assert.equal(normalizeFor('openai', {}).maxTokens, null);
});

test('normalizeFor reads json mode only from response_format json_object', () => {
  assert.equal(normalizeFor('openai', { response_format: { type: 'json_object' } }).jsonMode, true);
  assert.equal(normalizeFor('openai', { response_format: { type: 'text' } }).jsonMode, false);
  assert.equal(normalizeFor('openai', {}).jsonMode, false);
});

test('normalizeFor collects tools and skips one with no name', () => {
  const n = normalizeFor('openai', {
    tools: [
      { type: 'function', function: { name: 'lookup', description: 'd', parameters: { type: 'object' } } },
      { type: 'function', function: { description: 'no name' } },
      'junk',
    ],
  });
  assert.equal(n.tools.length, 1);
  assert.equal(n.tools[0].name, 'lookup');
  assert.equal(n.tools[0].description, 'd');
});

test('normalizeFor carries stop, temperature and top_p through', () => {
  const n = normalizeFor('openai', { stop: 'END', temperature: 0.2, top_p: 0.9 });
  assert.deepEqual(n.stops, ['END']);
  assert.equal(n.temperature, 0.2);
  assert.equal(n.topP, 0.9);
});

test('normalizeFor turns an image_url part into a base64 image spec', () => {
  const n = normalizeFor('openai', {
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'what is this' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,QUJD' } },
        ],
      },
    ],
  });
  assert.equal(n.userText, 'what is this');
  assert.deepEqual(n.image, { mediaType: 'image/png', data: 'QUJD' });
});

test('normalizeFor ignores an image part whose URL is not a data URL', () => {
  const n = normalizeFor('openai', {
    messages: [
      { role: 'user', content: [{ type: 'image_url', image_url: { url: 'https://x/a.png' } }] },
    ],
  });
  assert.equal(n.image, null);
});

test('normalizeFor ignores a non-user message when looking for the question', () => {
  const n = normalizeFor('openai', {
    messages: [
      { role: 'system', content: 'sys' },
      { role: 'assistant', content: 'I speak first' },
    ],
  });
  assert.equal(n.userText, '');
  assert.equal(n.systemText, 'sys');
});

test('normalizeFor tolerates a body with no messages at all', () => {
  const n = normalizeFor('openai', {});
  assert.equal(n.systemText, '');
  assert.equal(n.userText, '');
  assert.deepEqual(n.tools, []);
  assert.equal(n.image, null);
});

// ---------------------------------------------------------------------------
// stop reasons: provider vocabulary -> family -> client vocabulary
// ---------------------------------------------------------------------------

test('stopFamily reads each provider spelling of a normal end', () => {
  for (const reason of ['end_turn', 'endTurn', 'stop', 'stop_sequence', 'STOP']) {
    assert.equal(stopFamily(reason), 0, reason);
  }
});

test('stopFamily reads a token limit', () => {
  for (const reason of ['max_tokens', 'maxTokens', 'length', 'max_output_tokens', 'MAX_TOKENS']) {
    assert.equal(stopFamily(reason), 1, reason);
  }
});

test('stopFamily reads a tool call', () => {
  for (const reason of ['tool_use', 'toolUse', 'tool_calls', 'function_call']) {
    assert.equal(stopFamily(reason), 2, reason);
  }
});

test('stopFamily reads a refusal or a filter', () => {
  for (const reason of ['refusal', 'content_filter', 'contentFilter', 'safety', 'recitation']) {
    assert.equal(stopFamily(reason), 3, reason);
  }
});

test('stopFamily returns null for a reason it does not know', () => {
  assert.equal(stopFamily('something_new'), null);
  assert.equal(stopFamily(''), null);
});

test('the client vocabulary maps every family back', () => {
  assert.equal(openAIStop(0), 'stop');
  assert.equal(openAIStop(1), 'length');
  assert.equal(openAIStop(2), 'tool_calls');
  assert.equal(openAIStop(3), 'content_filter');
  assert.equal(anthropicStop(0), 'end_turn');
  assert.equal(anthropicStop(1), 'max_tokens');
  assert.equal(anthropicStop(2), 'tool_use');
  assert.equal(anthropicStop(3), 'refusal');
});

test('an unknown stop reason falls back to a normal end, not to an error', () => {
  assert.equal(openAIStop(stopFamily('something_new')), 'stop');
  assert.equal(anthropicStop(stopFamily('something_new')), 'end_turn');
});

test('a stop reason survives a round trip through every wire', () => {
  for (const reason of ['stop', 'max_tokens', 'tool_calls', 'content_filter']) {
    const family = stopFamily(reason);
    assert.equal(stopFamily(openAIStop(family) as string), family, reason);
    assert.equal(stopFamily(anthropicStop(family) as string), family, reason);
  }
});

// ---------------------------------------------------------------------------
// parsing an answer, per wire
// ---------------------------------------------------------------------------

test('parseAnswerFor reads an OpenAI answer with usage and tool calls', () => {
  const a = parseAnswerFor('openai', {
    choices: [
      {
        message: {
          content: 'hi',
          tool_calls: [{ id: 'call_1', function: { name: 'lookup', arguments: '{"q":"x"}' } }],
        },
        finish_reason: 'tool_calls',
      },
    ],
    usage: { prompt_tokens: 11, completion_tokens: 7 },
  });
  assert.equal(a.text, 'hi');
  assert.equal(a.stop, 'tool_calls');
  assert.equal(a.input, 11);
  assert.equal(a.output, 7);
  assert.equal(a.toolCalls.length, 1);
  assert.equal(a.toolCalls[0].name, 'lookup');
  assert.equal(a.toolCalls[0].arguments, '{"q":"x"}');
});

test('parseAnswerFor gives an OpenAI tool call a fallback id and empty arguments', () => {
  const a = parseAnswerFor('openai', {
    choices: [{ message: { tool_calls: [{ function: { name: 'x' } }] } }],
  });
  assert.equal(a.toolCalls[0].id, 'call_0');
  assert.equal(a.toolCalls[0].arguments, '{}');
});

test('parseAnswerFor reads an Anthropic answer and joins its text blocks', () => {
  const a = parseAnswerFor('anthropic', {
    content: [
      { type: 'text', text: 'part one ' },
      { type: 'tool_use', id: 'toolu_9', name: 'lookup', input: { q: 'x' } },
      { type: 'text', text: 'part two' },
    ],
    stop_reason: 'tool_use',
    usage: { input_tokens: 3, output_tokens: 4 },
  });
  assert.equal(a.text, 'part one part two');
  assert.equal(a.stop, 'tool_use');
  assert.equal(a.input, 3);
  assert.equal(a.output, 4);
  assert.equal(a.toolCalls[0].id, 'toolu_9');
  assert.deepEqual(JSON.parse(a.toolCalls[0].arguments), { q: 'x' });
});

test('parseAnswerFor reads a Gemini answer with its own usage field names', () => {
  const a = parseAnswerFor('gemini', {
    candidates: [
      {
        content: {
          parts: [{ text: 'hello ' }, { text: 'world' }, { functionCall: { name: 'lookup', args: { q: 'x' } } }],
        },
        finishReason: 'STOP',
      },
    ],
    usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 6 },
  });
  assert.equal(a.text, 'hello world');
  assert.equal(a.stop, 'STOP');
  assert.equal(a.input, 5);
  assert.equal(a.output, 6);
  assert.equal(a.toolCalls[0].name, 'lookup');
});

test('every wire parses an empty body without throwing', () => {
  for (const wire of ['openai', 'anthropic', 'gemini'] as const) {
    const a = parseAnswerFor(wire, {});
    assert.equal(a.text, '');
    assert.deepEqual(a.toolCalls, []);
    assert.equal(a.input, null);
    assert.equal(a.output, null);
  }
});

test('a wire parses a body full of wrong types without throwing', () => {
  const junk = { choices: 'no', content: 7, candidates: {}, usage: 'x', usageMetadata: [] };
  for (const wire of ['openai', 'anthropic', 'gemini'] as const) {
    assert.doesNotThrow(() => parseAnswerFor(wire, junk));
  }
});

// ---------------------------------------------------------------------------
// the routing contract
// ---------------------------------------------------------------------------

test('meetsRequires lets an unconstrained request through', () => {
  const row = rating('m', 'p', 100) as never;
  assert.equal(
    meetsRequires(row, { tools: false, vision: false, json: false, minContext: null }),
    true,
  );
});

test('meetsRequires rejects a model that lacks each capability in turn', () => {
  const requires = { tools: false, vision: false, json: false, minContext: null };
  assert.equal(meetsRequires(rating('m', 'p', 1, { tools: false }) as never, { ...requires, tools: true }), false);
  assert.equal(meetsRequires(rating('m', 'p', 1, { vision: false }) as never, { ...requires, vision: true }), false);
  assert.equal(meetsRequires(rating('m', 'p', 1, { json: false }) as never, { ...requires, json: true }), false);
  assert.equal(meetsRequires(rating('m', 'p', 1, { tools: true }) as never, { ...requires, tools: true }), true);
});

test('meetsRequires treats an unknown context length as too small', () => {
  const requires = { tools: false, vision: false, json: false, minContext: 8000 };
  assert.equal(meetsRequires(rating('m', 'p', 1, { contextLength: null }) as never, requires), false);
  assert.equal(meetsRequires(rating('m', 'p', 1, { contextLength: 7999 }) as never, requires), false);
  assert.equal(meetsRequires(rating('m', 'p', 1, { contextLength: 8000 }) as never, requires), true);
});

test('candidatesFor offers only free models', () => {
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('paid', 'alpha', 999, { free: false }), rating('free', 'alpha', 1)],
  });
  const got = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  assert.deepEqual(got.map((c) => c.modelId), ['free']);
});

test('candidatesFor ranks by rating, highest first', () => {
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('low', 'alpha', 10), rating('high', 'alpha', 900), rating('mid', 'alpha', 500)],
  });
  const got = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  assert.deepEqual(got.map((c) => c.modelId), ['high', 'mid', 'low']);
});

test('candidatesFor breaks a rating tie by provider id, then model id', () => {
  load({
    providers: [provider('zeta', 'openai'), provider('alpha', 'openai')],
    ratings: [rating('m', 'zeta', 100), rating('b', 'alpha', 100), rating('a', 'alpha', 100)],
  });
  const got = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  assert.deepEqual(
    got.map((c) => `${c.providerId}/${c.modelId}`),
    ['alpha/a', 'alpha/b', 'zeta/m'],
  );
});

test('candidatesFor excludes the Anthropic wire only when the body carries response_format', () => {
  load({
    providers: [provider('claude', 'anthropic'), provider('gpt', 'openai')],
    ratings: [rating('c1', 'claude', 900), rating('g1', 'gpt', 100)],
  });
  const requires = { tools: false, vision: false, json: false, minContext: null };
  assert.deepEqual(candidatesFor(requires, false).map((c) => c.modelId), ['c1', 'g1']);
  assert.deepEqual(candidatesFor(requires, true).map((c) => c.modelId), ['g1']);
});

test('candidatesFor offers nothing when no provider answered its probe', () => {
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('m', 'alpha', 100)],
  });
  state.anyDialable = true;
  state.probes.set('alpha', { ok: false, models: [] } as never);
  assert.deepEqual(candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false), []);
});

test('candidatesFor still tries every provider when not one probe answered', () => {
  // If nothing dialled, the probe is what is broken; refusing to route would
  // fail the whole workload for a fault on our side of the handshake.
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('m', 'alpha', 100)],
  });
  state.anyDialable = false;
  state.probes.set('alpha', { ok: false, models: [] } as never);
  assert.equal(candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false).length, 1);
});

test('capacityOf believes an unproven provider has room for three calls', () => {
  resetState();
  assert.equal(capacityOf('fresh'), 3);
});

test('capacityOf grows with successes and shrinks with plain refusals', () => {
  resetState();
  const s = providerStateOf('p');
  s.successes = 5;
  assert.equal(capacityOf('p'), 8);
  s.soft = 2;
  assert.equal(capacityOf('p'), 6);
});

test('capacityOf never goes below zero', () => {
  resetState();
  const s = providerStateOf('p');
  s.soft = 99;
  assert.equal(capacityOf('p'), 0);
});

test('capacityOf is zero for a provider that said it is out', () => {
  resetState();
  const s = providerStateOf('p');
  s.successes = 10;
  s.hard = 1;
  assert.equal(capacityOf('p'), 0);
});

test('attemptOrder puts the planned model first', () => {
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('best', 'alpha', 900), rating('worst', 'alpha', 1)],
  });
  const cands = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  const s = spec('r1');
  const plan = planRemaining([s as never], new Map());
  const order = attemptOrder(s as never, cands, plan);
  const planned = plan.target.get('r1');
  assert.equal(pairKey(order[0]), pairKey(planned as never));
});

test('attemptOrder puts a written-off provider last, never first', () => {
  load({
    providers: [provider('alpha', 'openai'), provider('beta', 'openai')],
    ratings: [rating('a1', 'alpha', 900), rating('b1', 'beta', 100)],
  });
  providerStateOf('alpha').hard = 1;
  const cands = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  const order = attemptOrder(spec('r1') as never, cands, { target: new Map() });
  assert.equal(order[0].providerId, 'beta');
  assert.equal(order[order.length - 1].providerId, 'alpha');
});

test('attemptOrder never offers a pair the provider called illegal', () => {
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('bad', 'alpha', 900), rating('good', 'alpha', 100)],
  });
  const cands = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  state.badPairs.add(pairKey({ providerId: 'alpha', modelId: 'bad' } as never));
  const order = attemptOrder(spec('r1') as never, cands, { target: new Map() });
  assert.deepEqual(order.map((c) => c.modelId), ['good']);
});

test('attemptOrder lists each candidate once', () => {
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('m', 'alpha', 100)],
  });
  const cands = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  const order = attemptOrder(spec('r1') as never, cands, { target: new Map() });
  assert.equal(order.length, new Set(order.map(pairKey)).size);
});

test('attemptOrder falls back to the least-used provider on a rating tie', () => {
  load({
    providers: [provider('alpha', 'openai'), provider('beta', 'openai')],
    ratings: [rating('a1', 'alpha', 500), rating('b1', 'beta', 500)],
  });
  providerStateOf('alpha').calls = 4;
  providerStateOf('beta').calls = 0;
  const cands = candidatesFor({ tools: false, vision: false, json: false, minContext: null }, false);
  const order = attemptOrder(spec('r1') as never, cands, { target: new Map() });
  assert.equal(order[0].providerId, 'beta');
});

test('planRemaining spreads near-equal ratings instead of draining one provider', () => {
  load({
    providers: [provider('alpha', 'openai'), provider('beta', 'openai')],
    ratings: [rating('a1', 'alpha', 500), rating('b1', 'beta', 500)],
  });
  const specs = [spec('r1'), spec('r2'), spec('r3'), spec('r4'), spec('r5')];
  const plan = planRemaining(specs as never[], new Map());
  const used = new Set(
    specs.map((s) => plan.target.get(s.id as string)?.providerId).filter(Boolean),
  );
  assert.equal(used.size, 2);
});

test('planRemaining serves the best rating first when capacity allows', () => {
  load({
    providers: [provider('alpha', 'openai'), provider('beta', 'openai')],
    ratings: [rating('a1', 'alpha', 900), rating('b1', 'beta', 100)],
  });
  const specs = [spec('r1'), spec('r2')];
  const plan = planRemaining(specs as never[], new Map());
  assert.equal(plan.target.get('r1')?.providerId, 'alpha');
});

test('planRemaining targets every request when a model can carry it', () => {
  load({
    providers: [provider('alpha', 'openai')],
    ratings: [rating('m', 'alpha', 500)],
  });
  const specs = [spec('r1'), spec('r2'), spec('r3')];
  const plan = planRemaining(specs as never[], new Map());
  assert.equal(plan.target.size, 3);
});

test('planRemaining leaves a request untargeted when nothing can serve it', () => {
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const plan = planRemaining([spec('r1') as never], new Map());
  assert.equal(plan.target.has('r1'), false);
});

// ---------------------------------------------------------------------------
// refusals: permanent or just this call
// ---------------------------------------------------------------------------

test('classifyRefusal reads a spent budget as permanent', () => {
  const r = classifyRefusal(429, { error: { code: 'quota', message: 'free budget is spent' } });
  assert.equal(r.hard, true);
  assert.equal(r.code, 'quota');
});

test('classifyRefusal reads "is out" as permanent', () => {
  assert.equal(classifyRefusal(503, { error: { message: 'this provider is out for now' } }).hard, true);
});

test('classifyRefusal treats a server error, a bad key and a forbidden key as permanent', () => {
  for (const status of [500, 502, 503, 401, 403]) {
    assert.equal(classifyRefusal(status, {}).hard, true, String(status));
  }
});

test('classifyRefusal treats a rate limit as just this call', () => {
  assert.equal(classifyRefusal(429, { error: { message: 'slow down' } }).hard, false);
});

test('classifyRefusal treats a bad request as just this call', () => {
  assert.equal(classifyRefusal(400, { error: { code: 'invalid_model' } }).hard, false);
});

test('classifyRefusal reads the code from code, then type, then status', () => {
  assert.equal(classifyRefusal(400, { error: { code: 'a' } }).code, 'a');
  assert.equal(classifyRefusal(400, { error: { type: 'b' } }).code, 'b');
  assert.equal(classifyRefusal(400, { error: { status: 'c' } }).code, 'c');
  assert.equal(classifyRefusal(400, {}).code, '');
});

test('classifyRefusal never throws on a body it cannot read', () => {
  for (const body of [null, 'text', 42, [], { error: 'not an object' }]) {
    assert.doesNotThrow(() => classifyRefusal(400, body));
  }
});

// ---------------------------------------------------------------------------
// inferRequires: what a request needs, from its own body
// ---------------------------------------------------------------------------

test('inferRequires sees tools, an image and json mode in an OpenAI body', () => {
  const r = inferRequires(
    {
      tools: [{ type: 'function', function: { name: 'x' } }],
      response_format: { type: 'json_object' },
      messages: [
        { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,QQ==' } }] },
      ],
    } as Json,
    'openai',
  );
  assert.deepEqual(r, { tools: true, vision: true, json: true, minContext: null });
});

test('inferRequires asks for nothing from a plain text request', () => {
  const r = inferRequires({ messages: [{ role: 'user', content: 'hello' }] } as Json, 'openai');
  assert.deepEqual(r, { tools: false, vision: false, json: false, minContext: null });
});

test('inferRequires reads an Anthropic body through the Anthropic normaliser', () => {
  const r = inferRequires(
    {
      system: 'sys',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'look' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'QQ==' } },
          ],
        },
      ],
    } as Json,
    'anthropic',
  );
  assert.equal(r.vision, true);
});

// ---------------------------------------------------------------------------
// upstreamBody: the client's request in the provider's own wire
// ---------------------------------------------------------------------------

test('upstreamBody carries the system text and the question to an [OI] provider', () => {
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const n = normalizeFor('openai', {
    messages: [
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
    ],
    max_tokens: 50,
  });
  const body = upstreamBody('openai', 'm1', n) as Json;
  const messages = body.messages as Json[];
  assert.equal(messages[0].role, 'system');
  assert.equal(messages[0].content, 'be brief');
  assert.equal(messages[1].role, 'user');
  assert.deepEqual(messages[1].content, [{ type: 'text', text: 'hi' }]);
  assert.equal(body.max_tokens, 50);
  assert.equal(body.model, 'm1');
  assert.equal(body.stream, false);
});

test('upstreamBody leaves out a limit the client did not send', () => {
  // Never invent max_tokens: the client's absence must stay an absence.
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const n = normalizeFor('openai', { messages: [{ role: 'user', content: 'hi' }] });
  for (const wire of ['openai', 'anthropic', 'gemini'] as const) {
    const body = upstreamBody(wire, 'm', n) as Json;
    assert.equal(body.max_tokens, undefined, wire);
    assert.equal(body.generationConfig, undefined, wire);
  }
});

test('upstreamBody translates the stop list into each wire\'s own field name', () => {
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const n = normalizeFor('openai', { messages: [{ role: 'user', content: 'hi' }], stop: ['END'] });
  assert.deepEqual((upstreamBody('openai', 'm', n) as Json).stop, ['END']);
  assert.deepEqual((upstreamBody('anthropic', 'm', n) as Json).stop_sequences, ['END']);
  assert.deepEqual(((upstreamBody('gemini', 'm', n) as Json).generationConfig as Json).stopSequences, ['END']);
});

test('upstreamBody gives each wire its own tool shape', () => {
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const n = normalizeFor('openai', {
    messages: [{ role: 'user', content: 'hi' }],
    tools: [{ type: 'function', function: { name: 'lookup', description: 'd', parameters: { type: 'object' } } }],
  });
  const oa = (upstreamBody('openai', 'm', n) as Json).tools as Json[];
  assert.equal(oa[0].type, 'function');
  assert.equal((oa[0].function as Json).name, 'lookup');

  const an = (upstreamBody('anthropic', 'm', n) as Json).tools as Json[];
  assert.equal(an[0].name, 'lookup');
  assert.deepEqual(an[0].input_schema, { type: 'object' });
  assert.equal(an[0].function, undefined);

  const gm = (upstreamBody('gemini', 'm', n) as Json).tools as Json[];
  const decl = (gm[0].functionDeclarations as Json[])[0];
  assert.equal(decl.name, 'lookup');
  assert.deepEqual(decl.parameters, { type: 'object' });
});

test('upstreamBody defaults a tool with no parameters to an empty object schema', () => {
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const n = normalizeFor('openai', {
    messages: [{ role: 'user', content: 'hi' }],
    tools: [{ type: 'function', function: { name: 'x' } }],
  });
  const oa = (upstreamBody('openai', 'm', n) as Json).tools as Json[];
  assert.deepEqual((oa[0].function as Json).parameters, { type: 'object', properties: {} });
});

test('upstreamBody asks Gemini for JSON through responseMimeType', () => {
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const n = normalizeFor('openai', {
    messages: [{ role: 'user', content: 'hi' }],
    response_format: { type: 'json_object' },
  });
  const body = upstreamBody('gemini', 'm', n) as Json;
  assert.equal((body.generationConfig as Json).responseMimeType, 'application/json');
});

test('upstreamBody does not send response_format to the Anthropic wire', () => {
  load({ providers: [provider('alpha', 'anthropic')], ratings: [] });
  const n = normalizeFor('openai', {
    messages: [{ role: 'user', content: 'hi' }],
    response_format: { type: 'json_object' },
  });
  assert.equal((upstreamBody('anthropic', 'm', n) as Json).response_format, undefined);
});

test('upstreamBody names the model in the body only where the wire wants it there', () => {
  load({ providers: [provider('alpha', 'openai')], ratings: [] });
  const n = normalizeFor('openai', { messages: [{ role: 'user', content: 'hi' }] });
  assert.equal((upstreamBody('openai', 'm1', n) as Json).model, 'm1');
  assert.equal((upstreamBody('anthropic', 'c1', n) as Json).model, 'c1');
  // Gemini names the model in the URL, not the body
  assert.equal((upstreamBody('gemini', 'g1', n) as Json).model, undefined);
});

test('openAIUserParts sends the image as a data URL next to the text', () => {
  const n = normalizeFor('openai', {
    messages: [{ role: 'user', content: [{ type: 'text', text: 'look' }] }],
  });
  n.image = { mediaType: 'image/png', data: 'QUJD' };
  assert.deepEqual(openAIUserParts(n), [
    { type: 'text', text: 'look' },
    { type: 'image_url', image_url: { url: 'data:image/png;base64,QUJD' } },
  ]);
});

test('an image with no text sends the image alone, not an empty text part', () => {
  const n = normalizeFor('openai', { messages: [{ role: 'user', content: '' }] });
  n.image = { mediaType: 'image/png', data: 'QUJD' };
  assert.deepEqual(openAIUserParts(n).map((p) => p.type), ['image_url']);
  assert.deepEqual(anthropicUserBlocks(n).map((p) => p.type), ['image']);
  assert.equal(geminiParts(n).length, 1);
  assert.ok(geminiParts(n)[0].inlineData);
});

test('toolChoiceForAnthropic maps each choice to its Anthropic shape', () => {
  assert.deepEqual(toolChoiceForAnthropic('auto'), { type: 'auto' });
  assert.deepEqual(toolChoiceForAnthropic('required'), { type: 'any' });
  assert.deepEqual(toolChoiceForAnthropic('any'), { type: 'any' });
  assert.deepEqual(toolChoiceForAnthropic({ function: { name: 'x' } }), { type: 'tool', name: 'x' });
});

test('toolChoiceForAnthropic has no equivalent for "none"', () => {
  assert.equal(toolChoiceForAnthropic('none'), null);
  assert.equal(toolChoiceForAnthropic('nonsense'), null);
});

test('upstreamBody gives an Anthropic provider its system field, not a system message', () => {
  load({ providers: [provider('claude', 'anthropic')], ratings: [] });
  const n = normalizeFor('openai', {
    messages: [
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
    ],
  });
  const body = upstreamBody('anthropic', 'c1', n) as Json;
  assert.equal(body.system, 'be brief');
  const messages = body.messages as Json[];
  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, 'user');
});

