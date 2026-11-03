import { readFile } from "node:fs/promises";
import { createGenerator } from "./generate.js";
import { ingestDirectory } from "./ingest.js";
import { SupportAssistant } from "./rag.js";
import { loadIndex } from "./store.js";

const DOCS_DIR = process.env.DOCS_DIR ?? "data/docs";
const INDEX_PATH = process.env.INDEX_PATH ?? "data/index.json";

interface EvalCase {
  question: string;
  /** Source document that should rank in the results, or null if the question is out of scope. */
  expectedSource: string | null;
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);

  if (command === "ingest") {
    const { documents, chunks } = await ingestDirectory(DOCS_DIR, INDEX_PATH);
    console.log(`Indexed ${chunks} chunks from ${documents} documents into ${INDEX_PATH}`);
    return;
  }

  if (command === "ask") {
    const question = rest.join(" ").trim();
    if (!question) throw new Error('Usage: npm run ask -- "your question"');
    const { embedder, store } = await loadIndex(INDEX_PATH);
    const assistant = new SupportAssistant(embedder, store, createGenerator());
    const result = await assistant.ask(question);
    console.log(`\n${result.answer}\n`);
    console.log(`grounded: ${result.grounded}  generator: ${result.generator}`);
    for (const s of result.sources) {
      console.log(`  - ${s.source} / ${s.heading}  (score ${s.score})`);
    }
    return;
  }

  if (command === "eval") {
    const cases = JSON.parse(await readFile("eval/questions.json", "utf8")) as EvalCase[];
    const { embedder, store } = await loadIndex(INDEX_PATH);
    const assistant = new SupportAssistant(embedder, store, createGenerator());

    let inScope = 0;
    let answered = 0;
    let top1 = 0;
    let top3 = 0;
    let outScope = 0;
    let refused = 0;
    for (const c of cases) {
      const hits = assistant.retrieve(c.question);
      const answer = await assistant.ask(c.question);
      if (c.expectedSource === null) {
        outScope += 1;
        if (!answer.grounded) refused += 1;
        console.log(`${answer.grounded ? "FAIL" : "ok  "}  [out of scope] ${c.question}  (best ${hits[0]?.score.toFixed(3)})`);
        continue;
      }
      inScope += 1;
      if (answer.grounded) answered += 1;
      const first = hits[0]?.chunk.source === c.expectedSource;
      const anyOf3 = hits.some((h) => h.chunk.source === c.expectedSource);
      if (first) top1 += 1;
      if (anyOf3) top3 += 1;
      console.log(`${first ? "ok  " : anyOf3 ? "top3" : "FAIL"}  ${c.question}  -> ${hits[0]?.chunk.source} (${hits[0]?.score.toFixed(3)})`);
    }
    console.log("");
    console.log(`Retrieval top-1 hit rate: ${top1}/${inScope}`);
    console.log(`Retrieval top-3 hit rate: ${top3}/${inScope}`);
    console.log(`In-scope questions answered (not refused): ${answered}/${inScope}`);
    console.log(`Out-of-scope questions refused: ${refused}/${outScope}`);
    return;
  }

  console.log("Usage: npm run ingest | npm run ask -- \"question\" | npm run eval | npm run serve");
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
