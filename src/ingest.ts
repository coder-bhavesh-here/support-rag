import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { chunkMarkdown, type Chunk } from "./chunk.js";
import { HashingEmbedder } from "./embed.js";
import { VectorStore, saveIndex } from "./store.js";

/** The text that gets embedded: heading plus body, so section titles help retrieval. */
export function embeddingText(chunk: Chunk): string {
  return chunk.heading ? `${chunk.heading}\n${chunk.text}` : chunk.text;
}

export async function ingestDirectory(docsDir: string, indexPath: string) {
  const files = (await readdir(docsDir)).filter((f) => f.endsWith(".md")).sort();
  const chunks: Chunk[] = [];
  for (const file of files) {
    const markdown = await readFile(join(docsDir, file), "utf8");
    chunks.push(...chunkMarkdown(file, markdown));
  }
  if (chunks.length === 0) throw new Error(`No markdown files found in ${docsDir}`);

  const embedder = new HashingEmbedder();
  const texts = chunks.map(embeddingText);
  embedder.fit(texts);
  const vectors = embedder.embed(texts);

  const store = new VectorStore();
  chunks.forEach((chunk, i) => store.add(chunk, vectors[i] as number[]));
  await saveIndex(indexPath, embedder, store);
  return { documents: files.length, chunks: chunks.length };
}
