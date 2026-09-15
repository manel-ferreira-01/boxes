/** Resolve serialized field values (see webui/core/serialize.py) into
 *  browser-usable numbers/images.  Nothing here knows any box by name. */

import type { SerRef, SerValue } from "./api"; // eslint noUnused: SerValue used in signatures

export const PALETTE = [
  "#5ba8ff", "#e0b15d", "#e05d5d", "#3fb96f", "#c9a6ff",
  "#5dd9c3", "#ff8fa3", "#9ecbff", "#b7e3a1", "#e8c07a",
];

export type RefKind = "array" | "buffer" | "file";

export function isRef(v: unknown): v is SerRef {
  return !!v && typeof v === "object" && !Array.isArray(v) &&
    ["array", "buffer", "file"].includes((v as SerRef).kind);
}

export function refKind(v: SerRef): RefKind {
  const k = (v as { kind?: string }).kind;
  return k === "array" || k === "buffer" || k === "file" ? (k as RefKind) : "file";
}

export function refUrl(v: unknown): string | null {
  if (isRef(v)) {
    const u = (v as { url?: string }).url;
    return typeof u === "string" ? u : null;
  }
  return null;
}

/** Inline numeric values (kind "array") as a flat number[]. */
export function inlineValues(v: unknown): number[] | null {
  if (!isRef(v)) return null;
  if (v.kind !== "array") return null;
  return flattenNumbers((v as { values?: unknown }).values);
}

export function flattenNumbers(v: unknown, out: number[] = []): number[] | null {
  if (typeof v === "number") {
    if (Number.isFinite(v)) { out.push(v); return out; }
    return null;
  }
  if (typeof v === "boolean") {
    out.push(v ? 1 : 0);
    return out;
  }
  if (Array.isArray(v)) {
    for (const x of v) {
      const r = flattenNumbers(x, out);
      if (r === null) return null;
    }
    return out;
  }
  return null;
}

/** Fetch a buffer artifact to a typed array (dtype-driven). */
export async function fetchTyped(
  v: SerValue,
): Promise<{ data: Float32Array | Float64Array | Uint8Array; shape: number[]; dtype: string } | null> {
  if (!isRef(v)) return inlineTyped(v as never);
  const kind = v.kind;
  const url = "url" in v ? v.url : undefined;
  const dtype = String((v as { dtype?: string }).dtype ?? "");
  const shape = ((v as { shape?: number[] }).shape ?? []).map(Number);
  if (kind === "array" || !url) return inlineTyped(v as never);
  const r = await fetch(url);
  if (!r.ok) return null;
  const buf = await r.arrayBuffer();
  return { data: typedFrom(dtype, buf), shape, dtype };
}

function typedFrom(dtype: string, buf: ArrayBuffer): Float32Array | Float64Array | Uint8Array {
  if (dtype === "float64") return new Float64Array(buf);
  if (dtype === "bool" || dtype === "uint8" || dtype === "") return new Uint8Array(buf);
  return new Float32Array(buf);
}

function inlineTyped(v: unknown): { data: Float32Array; shape: number[]; dtype: string } | null {
  if (!isRef(v)) return null;
  if (v.kind !== "array") return null;
  const flat = flattenNumbers((v as { values?: unknown }).values);
  const shape = ((v as { shape?: unknown[] }).shape ?? []).map(Number);
  if (!flat || flat.length === 0) {
    return { data: new Float32Array(0), shape, dtype: String((v as { dtype?: string }).dtype ?? "float") };
  }
  return { data: new Float32Array(flat), shape, dtype: String((v as { dtype?: string}).dtype ?? "float") };
}

/** Head preview of a (possibly remote) numeric value. */
export async function headOf(v: unknown, n: number): Promise<number[] | null> {
  if (isRef(v)) {
    const size = (v as { size?: number }).size;
    if (typeof size === "number" && size > 32 * 1024 * 1024) return null; // too big: skip preview
    const t = await fetchTyped(v as SerValue);
    return t ? Array.from(t.data).slice(0, n) : null;
  }
  const flat = flattenNumbers(v);
  return flat ? flat.slice(0, n) : null;
}

/** Stats over a numeric value (cheap path inline; bounded fetch otherwise). */
export async function statsOf(v: unknown): Promise<{
  min: number; max: number; mean: number; n: number;
} | null> {
  if (isRef(v)) {
    const size = (v as { size?: number }).size;
    if (typeof size === "number" && size > 64 * 1024 * 1024) return null;
    const t = await fetchTyped(v as SerValue);
    if (!t || t.data.length === 0) return null;
    const n = t.data.length;
    let min = Infinity, max = -Infinity, sum = 0;
    const d = t.data;
    for (let i = 0; i < n; i++) {
      const x = d[i];
      if (x < min) min = x;
      if (x > max) max = x;
      sum += x;
    }
    return { min, max, mean: sum / n, n };
  }
  const flat = flattenNumbers(v);
  if (!flat || flat.length === 0) return null;
  let min = Infinity, max = -Infinity, sum = 0;
  for (const x of flat) {
    if (x < min) min = x;
    if (x > max) max = x;
    sum += x;
  }
  return { min, max, mean: sum / flat.length, n: flat.length };
}

/** shape of a nested JSON array (best effort) — mirrors shapeOf on the console. */
export function inlineShape(values: unknown): number[] {
  const out: number[] = [];
  let cur: unknown = values;
  while (Array.isArray(cur) && cur.length > 0) {
    out.push(cur.length);
    cur = cur[0];
  }
  return out;
}

/** Per-item numeric resolution for result values that are a *list of items*
 *  (one dict per input image): pick ``item[prop]`` (or the value itself)
 *  and materialize the numbers — inline arrays stay inline, buffer
 *  artifacts are fetched only because the visualizer must render them.
 *  Non-numeric cells are skipped (visualizers degrade, never break). */
export async function itemNumerics(
  v: unknown, prop?: string,
): Promise<{ data: ArrayLike<number>; shape: number[]; url: string | null }[]> {
  const items: unknown[] = Array.isArray(v) ? v : [v];
  const out: { data: ArrayLike<number>; shape: number[]; url: string | null }[] = [];
  for (const it of items) {
    const cell = prop && it && typeof it === "object" && !Array.isArray(it)
      ? (it as Record<string, unknown>)[prop]
      : it;
    if (cell === undefined || cell === null) continue;
    const url = refUrl(cell);
    if (isRef(cell)) {
      const t = await fetchTyped(cell as SerValue);
      if (t && t.data.length > 0) out.push({ data: t.data, shape: t.shape, url });
      continue;
    }
    const flat = flattenNumbers(cell);
    if (flat && flat.length > 0) {
      out.push({ data: flat, shape: inlineShape(cell), url: null });
    }
  }
  return out;
}

export function shapeStr(shape: unknown): string {
  if (!Array.isArray(shape) || shape.length === 0) return "scalar";
  return `[${shape.join(", ")}]`;
}

export function downloadHref(v: unknown): string | null {
  const u = refUrl(v);
  return u;
}
