import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import type { Chunk } from "./chunk.js";
import { HashingEmbedder, dot, type EmbedderState } from "./embed.js";

export interface StoredChunk extends Chunk {
  vector: number[];
}

export interface SearchHit {
  chunk: Chunk;
  score: number;
}

/** Hashed vectors are mostly zeros, so they are saved as index/value pairs. */
interface SparseVector {
  i: number[];
  v: number[];
}

interface IndexFile {
  version: 1;
  embedder: EmbedderState;
  chunks: (Chunk & { vector: SparseVector })[];
}

function toSparse(vector: number[]): SparseVector {
  const i: number[] = [];
  const v: number[] = [];
  vector.forEach((x, idx) => {
    if (x !== 0) {
      i.push(idx);
      // 5 decimals is plenty; it does not change rankings in practice.
      v.push(Math.round(x * 1e5) / 1e5);
    }
  });
  return { i, v };
}

function toDense(sparse: SparseVector, dim: number): number[] {
  const vector = new Array<number>(dim).fill(0);
  sparse.i.forEach((idx, n) => {
    vector[idx] = sparse.v[n] as number;
  });
  return vector;
}

/** In-memory vector store with brute-force cosine search. Fine up to tens of thousands of chunks. */
export class VectorStore {
  constructor(private chunks: StoredChunk[] = []) {}

  get size(): number {
    return this.chunks.length;
  }

  add(chunk: Chunk, vector: number[]): void {
    this.chunks.push({ ...chunk, vector });
  }

  search(queryVector: number[], k: number): SearchHit[] {
    return this.chunks
      .map((c) => ({ chunk: stripVector(c), score: dot(queryVector, c.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  toJSON(): StoredChunk[] {
    return this.chunks;
  }
}

function stripVector(c: StoredChunk): Chunk {
  return { id: c.id, source: c.source, heading: c.heading, text: c.text };
}

export async function saveIndex(
  path: string,
  embedder: HashingEmbedder,
  store: VectorStore,
): Promise<void> {
  const file: IndexFile = {
    version: 1,
    embedder: embedder.state(),
    chunks: store.toJSON().map((c) => ({ ...c, vector: toSparse(c.vector) })),
  };
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(file));
}

export async function loadIndex(
  path: string,
): Promise<{ embedder: HashingEmbedder; store: VectorStore }> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch {
    throw new Error(`Index not found at ${path}. Run "npm run ingest" first.`);
  }
  const file = JSON.parse(raw) as IndexFile;
  if (file.version !== 1) throw new Error(`Unsupported index version: ${String(file.version)}`);
  const chunks = file.chunks.map((c) => ({ ...c, vector: toDense(c.vector, file.embedder.dim) }));
  return { embedder: new HashingEmbedder(file.embedder), store: new VectorStore(chunks) };
}
