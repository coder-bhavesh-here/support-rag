import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { pathToFileURL } from "node:url";
import { createGenerator } from "./generate.js";
import { SupportAssistant } from "./rag.js";
import { loadIndex } from "./store.js";

const MAX_BODY_BYTES = 10_000;
const MAX_QUESTION_CHARS = 500;

function send(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(payload) });
  res.end(payload);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const parts: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      parts.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(parts).toString("utf8")));
    req.on("error", reject);
  });
}

export function createApp(assistant: SupportAssistant) {
  return createServer(async (req, res) => {
    try {
      if (req.method === "GET" && req.url === "/health") {
        return send(res, 200, { status: "ok" });
      }
      if (req.method === "POST" && req.url === "/ask") {
        let body: unknown;
        try {
          body = JSON.parse(await readBody(req));
        } catch {
          return send(res, 400, { error: "Request body must be valid JSON under 10 KB." });
        }
        const question = (body as { question?: unknown }).question;
        if (typeof question !== "string" || question.trim().length === 0) {
          return send(res, 400, { error: 'Send JSON like {"question": "..."}.' });
        }
        if (question.length > MAX_QUESTION_CHARS) {
          return send(res, 400, { error: `Question must be under ${MAX_QUESTION_CHARS} characters.` });
        }
        return send(res, 200, await assistant.ask(question.trim()));
      }
      return send(res, 404, { error: "Not found. Try POST /ask or GET /health." });
    } catch {
      return send(res, 500, { error: "Internal error." });
    }
  });
}

async function main() {
  const indexPath = process.env.INDEX_PATH ?? "data/index.json";
  const port = Number(process.env.PORT ?? 3000);
  const { embedder, store } = await loadIndex(indexPath);
  const generator = createGenerator();
  const app = createApp(new SupportAssistant(embedder, store, generator));
  app.listen(port, () => {
    console.log(`support-rag listening on http://localhost:${port} (generator: ${generator.name})`);
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
