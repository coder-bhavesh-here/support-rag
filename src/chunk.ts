import { splitSentences } from "./text.js";

export interface Chunk {
  id: string;
  /** Source document name, e.g. "returns.md". */
  source: string;
  /** Nearest markdown heading, used for citations and as extra retrieval signal. */
  heading: string;
  text: string;
}

interface Section {
  heading: string;
  body: string;
}

function splitSections(markdown: string): Section[] {
  const sections: Section[] = [];
  let heading = "";
  let lines: string[] = [];
  const flush = () => {
    const body = lines.join("\n").trim();
    if (body) sections.push({ heading, body });
    lines = [];
  };
  for (const line of markdown.split(/\r?\n/)) {
    const m = /^#{1,3}\s+(.*\S)\s*$/.exec(line);
    if (m) {
      flush();
      heading = m[1] ?? "";
    } else {
      lines.push(line);
    }
  }
  flush();
  return sections;
}

/** Pack paragraphs into chunks of at most maxChars; split oversized paragraphs by sentence. */
function packParagraphs(body: string, maxChars: number): string[] {
  const pieces: string[] = [];
  for (const para of body.split(/\n\s*\n/)) {
    const p = para.trim();
    if (!p) continue;
    if (p.length <= maxChars) {
      pieces.push(p);
    } else {
      let current = "";
      for (const sentence of splitSentences(p)) {
        if (current && current.length + sentence.length + 1 > maxChars) {
          pieces.push(current);
          current = sentence;
        } else {
          current = current ? `${current} ${sentence}` : sentence;
        }
      }
      if (current) pieces.push(current);
    }
  }
  const chunks: string[] = [];
  let current = "";
  for (const piece of pieces) {
    if (current && current.length + piece.length + 2 > maxChars) {
      chunks.push(current);
      current = piece;
    } else {
      current = current ? `${current}\n\n${piece}` : piece;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

/**
 * Chunk a markdown document. Each section (by heading) is chunked separately so a
 * chunk never mixes topics, and every chunk keeps its heading for citations.
 */
export function chunkMarkdown(source: string, markdown: string, maxChars = 600): Chunk[] {
  const chunks: Chunk[] = [];
  let n = 0;
  for (const section of splitSections(markdown)) {
    for (const text of packParagraphs(section.body, maxChars)) {
      chunks.push({ id: `${source}#${n}`, source, heading: section.heading, text });
      n += 1;
    }
  }
  return chunks;
}
