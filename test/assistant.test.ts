import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { AnthropicGenerator, ExtractiveGenerator, createGenerator, type Generator } from "../src/generate.js";
import { ingestDirectory } from "../src/ingest.js";
import { FALLBACK_ANSWER, SupportAssistant } from "../src/rag.js";
import { createApp } from "../src/server.js";
import { loadIndex } from "../src/store.js";

let dir: string;
let embedder: Awaited<ReturnType<typeof loadIndex>>["embedder"];
let store: Awaited<ReturnType<typeof loadIndex>>["store"];

before(async () => {
  dir = await mkdtemp(join(tmpdir(), "support-rag-"));
  const indexPath = join(dir, "index.json");
  await ingestDirectory("data/docs", indexPath);
  ({ embedder, store } = await loadIndex(indexPath));
});

after(async () => {
  await rm(dir, { recursive: true, force: true });
});

test("extractive assistant answers from the docs and cites its sources", async () => {
  const assistant = new SupportAssistant(embedder, store, new ExtractiveGenerator());
  const result = await assistant.ask("How many days do I have to return an item?");
  assert.equal(result.grounded, true);
  assert.match(result.answer, /30 days/);
  assert.match(result.answer, /\[\d\]/);
  assert.equal(result.sources[0]?.source, "returns.md");
});

test("assistant refuses off-topic questions instead of guessing", async () => {
  const assistant = new SupportAssistant(embedder, store, new ExtractiveGenerator());
  const result = await assistant.ask("Who won the football match last night?");
  assert.equal(result.grounded, false);
  assert.equal(result.answer, FALLBACK_ANSWER);
});

test("a failing generator degrades to the extractive answer instead of erroring", async () => {
  const broken: Generator = {
    name: "broken",
    async generate() {
      throw new Error("model outage");
    },
  };
  const assistant = new SupportAssistant(embedder, store, broken);
  const result = await assistant.ask("When will I get my money back after a return?");
  assert.equal(result.grounded, true);
  assert.match(result.generator, /fallback/);
});

test("createGenerator uses Claude only when an API key is configured", () => {
  assert.equal(createGenerator({}).name, "extractive");
  assert.match(createGenerator({ ANTHROPIC_API_KEY: "k", ANTHROPIC_MODEL: "m" }).name, /^anthropic:m$/);
});

test("AnthropicGenerator sends the right request and parses the reply (mock server)", async () => {
  let seen: { headers: Record<string, unknown>; body: any } | undefined;
  const mock = createServer((req, res) => {
    const parts: Buffer[] = [];
    req.on("data", (c: Buffer) => parts.push(c));
    req.on("end", () => {
      seen = { headers: req.headers, body: JSON.parse(Buffer.concat(parts).toString("utf8")) };
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ content: [{ type: "text", text: "Returns take 30 days [1]." }] }));
    });
  });
  await new Promise<void>((r) => mock.listen(0, "127.0.0.1", r));
  try {
    const port = (mock.address() as AddressInfo).port;
    const generator = new AnthropicGenerator({ apiKey: "test-key", baseUrl: `http://127.0.0.1:${port}` });
    const assistant = new SupportAssistant(embedder, store, generator);
    const result = await assistant.ask("How many days do I have to return an item?");
    assert.equal(result.answer, "Returns take 30 days [1].");
    assert.equal(seen?.headers["x-api-key"], "test-key");
    assert.equal(seen?.headers["anthropic-version"], "2023-06-01");
    assert.match(seen?.body.system, /ONLY the numbered context/);
    assert.match(seen?.body.messages[0].content, /\[1\] \(returns\.md/);
    assert.match(seen?.body.messages[0].content, /Question: How many days/);
  } finally {
    mock.close();
  }
});

test("a non-200 reply from the model API falls back to the extractive answer", async () => {
  const mock = createServer((_req, res) => {
    res.writeHead(500);
    res.end("boom");
  });
  await new Promise<void>((r) => mock.listen(0, "127.0.0.1", r));
  try {
    const port = (mock.address() as AddressInfo).port;
    const generator = new AnthropicGenerator({ apiKey: "k", baseUrl: `http://127.0.0.1:${port}` });
    const result = await new SupportAssistant(embedder, store, generator).ask(
      "How many days do I have to return an item?",
    );
    assert.equal(result.grounded, true);
    assert.match(result.generator, /fallback/);
  } finally {
    mock.close();
  }
});

test("HTTP API: health, ask, and input validation", async () => {
  const app: Server = createApp(new SupportAssistant(embedder, store, new ExtractiveGenerator()));
  await new Promise<void>((r) => app.listen(0, "127.0.0.1", r));
  const base = `http://127.0.0.1:${(app.address() as AddressInfo).port}`;
  try {
    const health = await fetch(`${base}/health`);
    assert.equal(health.status, 200);

    const ok = await fetch(`${base}/ask`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: "Which payment cards do you take?" }),
    });
    assert.equal(ok.status, 200);
    const body = (await ok.json()) as { grounded: boolean; sources: { source: string }[] };
    assert.equal(body.grounded, true);
    assert.equal(body.sources[0]?.source, "orders-and-payments.md");

    const badJson = await fetch(`${base}/ask`, { method: "POST", body: "not json" });
    assert.equal(badJson.status, 400);

    const missing = await fetch(`${base}/ask`, { method: "POST", body: JSON.stringify({ q: "x" }) });
    assert.equal(missing.status, 400);

    const tooLong = await fetch(`${base}/ask`, {
      method: "POST",
      body: JSON.stringify({ question: "a".repeat(501) }),
    });
    assert.equal(tooLong.status, 400);

    const unknown = await fetch(`${base}/nope`);
    assert.equal(unknown.status, 404);
  } finally {
    app.close();
  }
});
