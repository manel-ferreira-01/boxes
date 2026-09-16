/** Tensor/array view: dtype/shape/size + stats + **.npy download** for
 *  both inline (small) arrays and buffer artifacts (large).
 *
 * Bandwidth rule: inline values (``kind: "array"``) already carry their
 * numbers in the API JSON, so stats + head preview are free.  Buffer
 * artifacts (``kind: "buffer"``) show metadata only — the payload is fetched
 * only when the user explicitly clicks "download raw" or ".npy". */
import { useEffect, useState } from "react";
import { isRef, shapeStr, statsOf, flattenNumbers } from "../resolvers";
import { fmtNum, bytesShort } from "../ui";

// ------------------------------------------------------------------ helpers

/** Flatten any nested number[] (or a single number[]) to a flat number[]. */
function walk(v: unknown, out: number[] = []): number[] {
  if (typeof v === "number" && Number.isFinite(v)) { out.push(v); return out; }
  if (Array.isArray(v)) for (const x of v) walk(x, out);
  return out;
}

/**
 * Write an IEEE 754 float32 .npy file (version 1.0, C-order) in JS and
 * trigger a browser download.
 *
 * Layout:
 *   magic (6 B) | version (2 B) | header_len (2 B, uint16 LE) | header | data
 * where header is  a Python dict string  +  spaces + \n,  padded so that
 *   10 + header_len_bytes ≡ 0 (mod 64)
 */
function saveNpy(flat: number[], shape: number[], filename: string) {
  const arr = new Float32Array(flat);
  const n   = arr.byteLength;
  const shapeStr =
    shape.length === 0
      ? "()"
      : "(" + shape.map(String).join(", ") + (shape.length === 1 ? "," : "") + ")";
  const base = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shapeStr}, }`;

  const PREFIX = 10;                              // 6 magic + 2 version + 2 header_len
  const minK   = Math.ceil((base.length + PREFIX) / 64);
  const target = 64 * minK - PREFIX;              // total header_len
  const pad    = target - base.length;
  const full   = base + " ".repeat(Math.max(0, pad - 1)) + "\n";
  const hb     = new TextEncoder().encode(full);

  // Sanity: 10 + hb.length must be a multiple of 64
  // (10 + hb.length = PREFIX + target = 64*minK ✓)

  const out = new Uint8Array(10 + hb.length + n);
  const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < 6; i++) out[i] = MAGIC[i];
  out[6] = 1; out[7] = 0;
  new DataView(out.buffer).setUint16(8, hb.length, true);  // header_len (LE)
  out.set(hb, 10);
  out.set(new Uint8Array(arr.buffer, arr.byteOffset, n), 10 + hb.length);

  const blob = new Blob([out.buffer], { type: "application/octet-stream" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ------------------------------------------------------------------ component

export function TensorView({ value, title }: { value: unknown; title?: string }) {
  const [stats, setStats]       = useState<{ min: number; max: number; mean: number; n: number } | null>(null);
  const [statsBusy, setStatsBusy] = useState(false);
  const [head, setHead]         = useState<number[] | null>(null);
  const [kind, setKind]         = useState<string | null>(null);
  const [dtype, setDtype]       = useState<string>("");
  const [shape, setShape]       = useState<string>("scalar");
  const [size, setSize]         = useState<number | null>(null);
  const [downloading, setDownloading] = useState(false);

  const isBuffer  = isRef(value) && (value as { kind?: string }).kind === "buffer";
  const isInline  = isRef(value) && (value as { kind?: string }).kind === "array";
  const shapeArr  = isRef(value) ? ((value as { shape?: number[] }).shape ?? []) : [];
  const shapeKey  = shapeArr.map(String).join("x") || "scalar";
  const npyFileName = `tensor_${shapeKey}.npy`;

  useEffect(() => {
    let alive = true;
    setStats(null); setHead(null); setSize(null);
    if (isRef(value)) {
      const r = value as { kind?: string; dtype?: string; shape?: number[]; size?: number };
      setKind(String(r.kind || ""));
      setDtype(String(r.dtype || ""));
      setShape(shapeStr(r.shape));
      if (typeof r.size === "number") setSize(r.size);
    }
    if (isBuffer) return () => { alive = false; };

    (async () => {
      setStatsBusy(true);
      try {
        const [s, h] = await Promise.all([
          statsOf(value),
          (async () => {
            if (value && typeof value === "object" && !Array.isArray(value) && "values" in (value as object)) {
              const flat: number[] = [];
              walk((value as { values: unknown }).values, flat);
              return flat.slice(0, 48);
            }
            return null;
          })(),
        ]);
        if (!alive) return;
        setStats(s);
        setHead(h);
      } finally {
        if (alive) setStatsBusy(false);
      }
    })();
    return () => { alive = false; };
  }, [value, isBuffer]);

  const url = isRef(value) ? (value as { url?: string }).url : null;

  async function handleDownloadNpy() {
    try {
      setDownloading(true);
      let flat: number[];
      if (isBuffer && url) {
        // large payload: fetch raw bytes, reinterpret as float32
        const r   = await fetch(url);
        const buf = new Float32Array(await r.arrayBuffer());
        flat      = Array.from(buf);
      } else if (isInline && value && typeof value === "object" && "values" in (value as object)) {
        const f: number[] = [];
        walk((value as { values: unknown }).values, f);
        flat = f;
      } else {
        const f = flattenNumbers(value);
        if (!f) return;
        flat = f;
      }
      saveNpy(flat, shapeArr, npyFileName);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      <div className="tensorhead">
        <span className="kv"><b>type</b>{kind === "array" ? "inline" : kind ?? "value"}</span>
        <span className="kv"><b>dtype</b>{dtype || "mixed"}</span>
        <span className="kv"><b>shape</b>{shape}</span>
        {size !== null && <span className="kv"><b>size</b>{bytesShort(size)}</span>}
        {stats && (
          <>
            <span className="kv"><b>min</b>{fmtNum(stats.min)}</span>
            <span className="kv"><b>max</b>{fmtNum(stats.max)}</span>
            <span className="kv"><b>mean</b>{fmtNum(stats.mean)}</span>
            <span className="kv"><b>n</b>{stats.n}</span>
          </>
        )}
        {(isBuffer || isInline) && (
          <button
            className="btn small"
            disabled={downloading}
            onClick={handleDownloadNpy}
            title="Download as .npy (float32, C-order)"
          >
            {downloading ? "…" : `↓ ${npyFileName}`}
          </button>
        )}
      </div>
      {isBuffer && (
        <div className="viz-note">
          large payload — metadata only; download with &ldquo;&darr; .npy&rdquo; when needed
        </div>
      )}
      {head && head.length > 0 && (
        <pre className="json" style={{ maxHeight: 160, overflow: "auto" }}>
          {head.map((x) => Number(x.toPrecision(5))).join(" ")}{head.length >= 48 ? " …" : ""}
        </pre>
      )}
      {statsBusy && <span className="spinnerbox"><span className="spinner" /></span>}
    </div>
  );
}
