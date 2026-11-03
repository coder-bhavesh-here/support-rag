import type { Embedder } from "./embed.js";
import { ExtractiveGenerator, type Generator } from "./generate.js";
import type { VectorStore } from "./store.js";

export interface Source {
  id: string;
  source: string;
  heading: string;
  score: number;
}

export interface Answer {
  answer: string;
  /** False when nothing in the knowledge base was relevant enough; hand off to a human. */
  grounded: boolean;
  sources: Source[];
  generator: string;
}

export interface AssistantOptions {
  topK?: number;
  /** Minimum cosine similarity of the best hit for the question to count as answerable. */
  minScore?: number;
}

export const FALLBACK_ANSWER =
  "I couldn't find that in our help documentation. Please contact support so a person can help.";

export class SupportAssistant {
  private topK: number;
  private minScore: number;

  constructor(
    private embedder: Embedder,
    private store: VectorStore,
    private generator: Generator,
    opts: AssistantOptions = {},
  ) {
    this.topK = opts.topK ?? 3;
    // Chosen from the eval set: answerable questions scored >= 0.068 and off-topic ones
    // <= 0.037. Re-tune it whenever the embedder or the documents change.
    this.minScore = opts.minScore ?? 0.05;
  }

  /** Retrieval only. Exposed separately so it can be evaluated without any generator. */
  retrieve(question: string) {
    const [queryVector] = this.embedder.embed([question]);
    return this.store.search(queryVector as number[], this.topK);
  }

  async ask(question: string): Promise<Answer> {
    const hits = this.retrieve(question);
    const sources: Source[] = hits.map((h) => ({
      id: h.chunk.id,
      source: h.chunk.source,
      heading: h.chunk.heading,
      score: Math.round(h.score * 1000) / 1000,
    }));

    if (hits.length === 0 || (hits[0]?.score ?? 0) < this.minScore) {
      return { answer: FALLBACK_ANSWER, grounded: false, sources, generator: "none" };
    }

    let generatorName = this.generator.name;
    let answer: string;
    try {
      answer = await this.generator.generate(question, hits);
    } catch {
      // A model outage should degrade the answer, not break support.
      const fallback = new ExtractiveGenerator();
      answer = await fallback.generate(question, hits);
      generatorName = `${fallback.name} (fallback)`;
    }
    if (!answer) {
      return { answer: FALLBACK_ANSWER, grounded: false, sources, generator: generatorName };
    }
    return { answer, grounded: true, sources, generator: generatorName };
  }
}
