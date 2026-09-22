import { promises as fs } from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";

const ROOTS = ["runtime", "futhark_kernels", "docs", "ci", "src/lib/superopt"];

async function walk(dir: string, base: string, out: string[]): Promise<void> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    const rel = path.relative(base, full);
    if (e.isDirectory()) await walk(full, base, out);
    else out.push(rel);
  }
}

function languageOf(file: string): string {
  if (file.endsWith(".zig")) return "zig";
  if (file.endsWith(".fut")) return "futhark";
  if (file.endsWith(".ts") || file.endsWith(".tsx")) return "typescript";
  if (file.endsWith(".md")) return "markdown";
  if (file.endsWith(".sh")) return "bash";
  if (file.endsWith(".txt")) return "zig";
  return "text";
}

export async function GET(request: Request) {
  const cwd = process.cwd();
  const url = new URL(request.url);
  const file = url.searchParams.get("file");

  if (!file) {
    const files: Array<{ path: string; bytes: number; language: string; root: string }> = [];
    for (const root of ROOTS) {
      const abs = path.join(cwd, root);
      const rels: string[] = [];
      await walk(abs, cwd, rels);
      for (const rel of rels) {
        const stat = await fs.stat(path.join(cwd, rel)).catch(() => null);
        if (!stat) continue;
        files.push({ path: rel, bytes: stat.size, language: languageOf(rel), root });
      }
    }
    files.sort((a, b) => a.path.localeCompare(b.path));
    return Response.json({ roots: ROOTS, files });
  }

  const normalized = path.normalize(file).replace(/^(\.\.[/\\])+/, "");
  const allowed = ROOTS.some((r) => normalized === r || normalized.startsWith(`${r}/`));
  if (!allowed || normalized.includes("..")) {
    return Response.json({ error: "path not allowed" }, { status: 403 });
  }
  try {
    const content = await fs.readFile(path.join(cwd, normalized), "utf8");
    return Response.json({ path: normalized, language: languageOf(normalized), content });
  } catch {
    return Response.json({ error: "file not found" }, { status: 404 });
  }
}
