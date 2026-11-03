const STOPWORDS = new Set(
  (
    "a an and are as at be by can do does for from has have how i if in is it its my of on or our so that the their " +
    "them then there these they this to was we what when where which who will with you your am been being but did " +
    "get got me not no than too very"
  ).split(" "),
);

/** Very small suffix stemmer: enough to match "returns"/"returned"/"returning". */
export function stem(word: string): string {
  if (word.length > 5 && word.endsWith("ing")) return word.slice(0, -3);
  if (word.length > 4 && word.endsWith("ed")) return word.slice(0, -2);
  if (word.length > 4 && word.endsWith("es")) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith("s") && !word.endsWith("ss")) return word.slice(0, -1);
  return word;
}

/** Lowercase, split into alphanumeric tokens, drop stopwords, stem. */
export function tokenize(text: string): string[] {
  const raw = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  const out: string[] = [];
  for (const w of raw) {
    if (STOPWORDS.has(w)) continue;
    out.push(stem(w));
  }
  return out;
}

/** Split text into sentences. Keeps it simple: split after . ! ? followed by whitespace. */
export function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
