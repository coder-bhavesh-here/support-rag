# support-rag

A small retrieval-augmented support assistant in TypeScript. It answers customer questions from a folder of help documents, cites where each answer came from, and refuses to guess when the documents don't cover the question.

It runs fully offline by default (no API key, no model download). Set an Anthropic API key and the same retrieval feeds Claude to write the answer instead.

> The help documents in `data/docs` describe a **fictional** car-audio store and are sample data only.

## Quick start

```bash
npm install
npm run ingest                                  # chunk + embed data/docs into data/index.json
npm run ask -- "How long does standard delivery take?"
npm run serve                                   # HTTP API on http://localhost:3000
npm run eval                                    # retrieval + refusal evaluation
npm test                                        # unit and API tests
```

Ask over HTTP:

```bash
curl -s -X POST localhost:3000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Can I return opened electronics?"}'
```

```json
{
  "answer": "Opened electronics can be returned within 30 days only if they are not installed and all accessories are included. [1]",
  "grounded": true,
  "sources": [{ "id": "returns.md#0", "source": "returns.md", "heading": "Return window", "score": 0.371 }],
  "generator": "extractive"
}
```

When nothing relevant is found, `grounded` is `false` and the answer tells the customer to contact support.

## How it works

1. **Chunk** (`src/chunk.ts`): each markdown file is split by heading, then packed by paragraph into chunks of at most 600 characters. A chunk never mixes sections and keeps its heading for citations.
2. **Embed** (`src/embed.ts`): an offline hashing embedder turns text into unit vectors. Unigrams and bigrams are hashed into 16,384 buckets with a random sign and weighted by sublinear term frequency and IDF learned from the corpus. The `Embedder` interface is small, so a neural embedding model can replace it.
3. **Store** (`src/store.ts`): an in-memory store with brute-force cosine search, saved to disk as sparse vectors (about 60 KB for the sample docs).
4. **Retrieve and decide** (`src/rag.ts`): take the top 3 chunks. If the best score is under `minScore`, refuse instead of answering.
5. **Generate** (`src/generate.ts`): by default, an extractive generator selects the sentences that best overlap the question and cites their chunks. With `ANTHROPIC_API_KEY` set, Claude writes the answer using only the numbered passages. If the model call fails, the assistant falls back to the extractive answer instead of returning an error.

## Using Claude for answers

```bash
export ANTHROPIC_API_KEY=your-key
export ANTHROPIC_MODEL=claude-sonnet-5-5   # optional; set any model you have access to
npm run serve
```

The prompt tells the model to answer only from the numbered context, cite passages, say it doesn't know when the context is silent, and treat the context as reference material rather than instructions. The request format and the fallback path are covered by tests against a local mock server. The tests do not call the live API.

## Evaluation

`npm run eval` runs 18 hand-written questions (15 answerable, 3 off-topic) from `eval/questions.json`:

| Metric | Result |
| --- | --- |
| Correct source ranked first | 13 / 15 |
| Correct source in top 3 | 14 / 15 |
| Answerable questions answered (not refused) | 15 / 15 |
| Off-topic questions refused | 3 / 3 |

Read these numbers with care: the set is small, and the embedding size (16,384) and refusal threshold (0.05) were chosen by looking at this same set, so the results are optimistic. With 1,024 buckets, hash collisions made off-topic questions score as high as on-topic ones; increasing the size fixed that.

## Known limitations

- **Lexical, not semantic.** The offline embedder matches words, not meaning. "How much does it cost to have a subwoofer installed?" misses the "Professional installation" section because the stemmer doesn't reduce "installation" to "install". A neural embedding model is the fix; the `Embedder` interface is where it plugs in.
- **Brute-force search.** Fine for thousands of chunks, not millions. Use an approximate-nearest-neighbour index or a vector database beyond that.
- **No authentication or rate limiting** on the HTTP server. It validates input size and shape only, so put it behind a gateway before exposing it.
- **The refusal threshold depends on the embedder and the documents.** Re-tune it whenever either changes.

## Project layout

```
src/text.ts      tokenizer, stemmer, sentence splitter
src/chunk.ts     markdown chunking
src/embed.ts     Embedder interface + offline hashing embedder
src/store.ts     vector store and index persistence
src/generate.ts  extractive and Anthropic generators
src/rag.ts       retrieve, decide, generate, fall back
src/server.ts    HTTP API (node:http, no framework)
src/cli.ts       ingest / ask / eval
test/            node:test suites
eval/            evaluation questions
data/docs/       sample (fictional) help documents
```

## License

MIT
