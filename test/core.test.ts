import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { chunkMarkdown } from "../src/chunk.js";
import { HashingEmbedder, dot } from "../src/embed.js";
import { ingestDirectory } from "../src/ingest.js";
import { loadIndex } from "../src/store.js";
import { stem, tokenize } from "../src/text.js";

test("tokenize drops stopwords and stems plurals and verb endings", () => {
  assert.deepEqual(tokenize("How do I return the items?"), ["return", "item"]);
  assert.equal(stem("returned"), "return");
  assert.equal(stem("returns"), "return");
  assert.equal(stem("class"), "class");
});

test("chunkMarkdown keeps headings, respects max size, and never mixes sections", () => {
  const md = [
    "# Title",
    "",
    "## Alpha",
    "",
    "First paragraph about alpha.",
    "",
    "## Beta",
    "",
    Array.from({ length: 30 }, (_, i) => `Beta sentence number ${i}.`).join(" "),
  ].join("\n");
  const chunks = chunkMarkdown("doc.md", md, 200);
  assert.ok(chunks.length >= 3);
  assert.ok(chunks.every((c) => c.text.length <= 200 && c.text.length > 0));
  assert.equal(chunks[0]?.heading, "Alpha");
  assert.ok(chunks.filter((c) => c.heading === "Beta").length >= 2);
  assert.ok(chunks.filter((c) => c.heading === "Alpha").every((c) => !c.text.includes("Beta")));
  assert.equal(new Set(chunks.map((c) => c.id)).size, chunks.length);
});

test("embeddings are unit length and related text scores higher than unrelated text", () => {
  const corpus = [
    "refund returns policy for unused items",
    "shipping times for standard delivery",
    "warranty covers manufacturing defects",
  ];
  const embedder = new HashingEmbedder();
  embedder.fit(corpus);
  const [query, related, unrelated] = embedder.embed([
    "how do I get a refund for a return",
    corpus[0] as string,
    corpus[1] as string,
  ]) as [number[], number[], number[]];
  assert.ok(Math.abs(dot(query, query) - 1) < 1e-9);
  assert.ok(dot(query, related) > dot(query, unrelated));
});

test("empty and stopword-only text embeds to a zero vector without crashing", () => {
  const embedder = new HashingEmbedder();
  const [vec] = embedder.embed(["the and of"]) as [number[]];
  assert.ok(vec.every((v) => v === 0));
});

test("ingest then load round-trips and returns the right source for a known question", async () => {
  const dir = await mkdtemp(join(tmpdir(), "support-rag-"));
  try {
    const indexPath = join(dir, "index.json");
    const { documents, chunks } = await ingestDirectory("data/docs", indexPath);
    assert.equal(documents, 6);
    assert.ok(chunks > 10);
    const { embedder, store } = await loadIndex(indexPath);
    assert.equal(store.size, chunks);
    const [q] = embedder.embed(["How do I reset my password?"]) as [number[]];
    const top = store.search(q, 1)[0];
    assert.equal(top?.chunk.source, "account-and-support.md");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("loadIndex gives a helpful error when the index is missing", async () => {
  await assert.rejects(() => loadIndex("does/not/exist.json"), /npm run ingest/);
});
