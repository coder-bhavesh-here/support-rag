import { tokenize } from "./text.js";

/** Anything that turns text into fixed-size, L2-normalised vectors. */
export interface Embedder {
  readonly dim: number;
  embed(texts: string[]): number[][];
}

function fnv1a(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export interface EmbedderState {
  dim: number;
  idf: Record<string, number>;
  defaultIdf: number;
}

/**
 * Offline embedder using the hashing trick: unigrams and bigrams are hashed into a
 * fixed-size vector with a random sign, weighted by sublinear term frequency and IDF
 * learned from the corpus. It gives real vector search without any API key or model
 * download. Swap in a neural embedding model by implementing the Embedder interface.
 */
export class HashingEmbedder implements Embedder {
  readonly dim: number;
  private idf: Record<string, number>;
  private defaultIdf: number;

  /**
   * 16384 buckets keeps hash collisions rare for a few thousand distinct terms. With only
   * 1024 buckets, unrelated words collided often enough that off-topic questions scored
   * as high as on-topic ones.
   */
  constructor(state: EmbedderState = { dim: 16384, idf: {}, defaultIdf: 1 }) {
    this.dim = state.dim;
    this.idf = state.idf;
    this.defaultIdf = state.defaultIdf;
  }

  private features(text: string): Map<string, number> {
    const tokens = tokenize(text);
    const counts = new Map<string, number>();
    const add = (f: string) => counts.set(f, (counts.get(f) ?? 0) + 1);
    for (let i = 0; i < tokens.length; i++) {
      add(tokens[i] as string);
      if (i + 1 < tokens.length) add(`${tokens[i]}_${tokens[i + 1]}`);
    }
    return counts;
  }

  /** Learn IDF weights from the corpus (one string per chunk). */
  fit(corpus: string[]): void {
    const df = new Map<string, number>();
    for (const doc of corpus) {
      for (const f of this.features(doc).keys()) df.set(f, (df.get(f) ?? 0) + 1);
    }
    const n = corpus.length;
    const idf: Record<string, number> = {};
    for (const [f, d] of df) idf[f] = Math.log((n + 1) / (d + 1)) + 1;
    this.idf = idf;
    this.defaultIdf = Math.log(n + 1) + 1;
  }

  embed(texts: string[]): number[][] {
    return texts.map((t) => this.embedOne(t));
  }

  private embedOne(text: string): number[] {
    const vec = new Array<number>(this.dim).fill(0);
    for (const [feature, tf] of this.features(text)) {
      const isBigram = feature.includes("_");
      const weight =
        (1 + Math.log(tf)) * (this.idf[feature] ?? this.defaultIdf) * (isBigram ? 0.5 : 1);
      const index = fnv1a(feature) % this.dim;
      const sign = fnv1a(`${feature}#`) & 1 ? 1 : -1;
      vec[index] = (vec[index] as number) + sign * weight;
    }
    return normalize(vec);
  }

  state(): EmbedderState {
    return { dim: this.dim, idf: this.idf, defaultIdf: this.defaultIdf };
  }
}

export function normalize(vec: number[]): number[] {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  return vec.map((v) => v / norm);
}

/** Cosine similarity for already-normalised vectors is just the dot product. */
export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}
