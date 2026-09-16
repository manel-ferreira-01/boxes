/** Tensor/array view: dtype/shape/size + stats + a **dtype-faithful .npy
 *  download** for both inline (small) arrays and buffer artifacts (large).
 *
 * Bandwidth rule: inline values (``kind: "array"``) already carry their numbers
 * in the API JSON, so stats + head preview are free.  Buffer artifacts
 * (``kind: "buffer"``) show metadata only — the payload is fetched only when
 * the user explicitly clicks "↓ .npy".
 *
 * The .npy is written with the array's **true dtype** (not assumed float32):
 * a float32 tensor → ``<f4``; a bool tensor (e.g. TAPNext ``visibles``) →
 * ``|b1`` (1-byte, 0/1); ints → their signed size.  This matters because a
 * bool/integer tensor mis-labelled as float would both load with the wrong
 * dtype and, for a bool inline array, lose every value. */
import { useEffect, useState } from "react";
import { isRef, shapeStr, statsOf } from "../resolvers";
import { fmtNum, bytesShort } from "../ui";

// ------------------------------------------------------------------ helpers

/** Collect numeric *and* boolean leaves into a flat number array in
 *  row-major order (bool → 0/1).  This is what lets a bool inline tensor
 *  (JSON booleans) round-trip into a proper 1-byte bool .npy. */
function collect(v: unknown, out: number[] = []): number[] {
  if (typeof v === "number" && Number.isFinite(v)) { out.push(v); return out; }
  if (typeof v === "boolean") { out.push(v ? 1 : 0); return out; }
  if (Array.isArray(v)) { for (const x of v) collect(x, out); return out; }
  return out;
}

/** dtype → numpy descr + itemsize (bytes per element).  These are the two
 *  facts we need to write a correct .npy header.  The concrete TypedArray
 *  is chosen by ``encodeFlat`` below when re-encoding an inline value. */
const DTYPE: Record<string, { descr: string; itemsize: number }> = {
  bool:      { descr: "|b1", itemsize: 1 },
  "int8":    { descr: "<i1", itemsize: 1 },
  "int16":   { descr: "<i2", itemsize: 2 },
  "int32":   { descr: "<i4", itemsize: 4 },
  "int64":   { descr: "<i8", itemsize: 8 },   // best-effort: encoded via float64
  "uint8":   { descr: "|u1", itemsize: 1 },
  "uint16":  { descr: "<u2", itemsize: 2 },
  "uint32":  { descr: "<u4", itemsize: 4 },
  "float16": { descr: "<f2", itemsize: 2 },   // best-effort: encoded via float32
  "float32": { descr: "<f4", itemsize: 4 },
  "float64": { descr: "<f8", itemsize: 8 },
};
function dtypeInfo(name: string) { return DTYPE[name] ?? DTYPE.float32; }

function flatShape(shape: number[]): number {
  let p = 1; for (const s of shape) p *= s; return p;
}

/** Re-encode a flat number[] (bool → 0/1 already folded by ``collect``)
 *  into the little-endian byte layout for the declared dtype.
 *
 *  int64 and float16 are not representable in a single JS numeric, so we
 *  store their values in the closest larger native array (int64 → float64,
 *  float16 → float32).  The header still declares the *original* dtype, and
 *  a size mismatch is logged so the consumer knows about it.  For the dtypes
 *  our boxes actually use (bool / float32) this is exact. */
function encodeFlat(flat: number[], dtypeName: string): Uint8Array {
  let ta: Uint8Array | Uint16Array | Uint32Array
      | Int8Array  | Int16Array | Int32Array
      | Float32Array | Float64Array;
  switch (dtypeName) {
    case "bool":
    case "uint8":   ta = new Uint8Array(flat);   break;
    case "int8":    ta = new Int8Array(flat);    break;
    case "uint16":  ta = new Uint16Array(flat);  break;
    case "int16":   ta = new Int16Array(flat);   break;
    case "uint32":  ta = new Uint32Array(flat);  break;
    case "int32":   ta = new Int32Array(flat);   break;
    case "int64":   ta = new Float64Array(flat); break;   // best-effort
    case "float64": ta = new Float64Array(flat); break;
    case "float16": ta = new Float32Array(flat); break;   // best-effort
    default:        ta = new Float32Array(flat); break;   // float32 / unknown
  }
  const expected = flat.length * dtypeInfo(dtypeName).itemsize;
  if (ta.byteLength !== expected) {
    console.warn(`[tensor] inline encode ${ta.byteLength} B != declared ${expected} B (dtype=${dtypeName})`);
  }
  return new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
}

/**
 * Write a little-endian .npy (v1.0, C-order) for an array of the given dtype
 * and trigger a browser download.  ``payload`` is the raw element bytes in
 * that dtype (a Uint8Array of 0/1 for bool, a Float32Array payload for
 * float32, …).  For buffer artifacts we pass the fetched payload verbatim so
 * no re-encode is needed.
 */
function saveNpy(payload: Uint8Array, dtypeName: string, shape: number[], filename: string) {
  const info = dtypeInfo(dtypeName);
  const n = payload.length;

  const tuple = shape.length === 0 ? "()" :
    "(" + shape.map(String).join(", ") + (shape.length === 1 ? "," : "") + ")";
  const base = `{'descr': '${info.descr}', 'fortran_order': False, 'shape': ${tuple}, }`;

  const PREFIX = 10;                       // 6 magic + 2 version + 2 header_len
  const minK   = Math.ceil((base.length + PREFIX) / 64);
  const target = 64 * minK - PREFIX;       // total header_len
  const full   = base + " ".repeat(Math.max(0, target - base.length - 1)) + "\n";
  const hb     = new TextEncoder().encode(full);
  // invariant: 10 + hb.length = 64*minK (a multiple of 64) — numpy requirement

  const out = new Uint8Array(PREFIX + hb.length + n);
  const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];   // \x93NUMPY
  for (let i = 0; i < 6; i++) out[i] = MAGIC[i];
  out[6] = 1; out[7] = 0;
  new DataView(out.buffer).setUint16(8, hb.length, true);
  out.set(hb, PREFIX);
  out.set(payload, PREFIX + hb.length);

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
  const rawRef    = isRef(value) ? (value as { dtype?: string; shape?: number[] }) : null;
  const dtypeName = rawRef?.dtype || "float32";          // declared by the serializer (bool / float32 / …)
  const shapeArr  = rawRef?.shape ?? [];
  const tupleKey  = shapeArr.length ? shapeArr.map(String).join("x") : "scalar";
  const npyFileName = `tensor_${tupleKey}_${dtypeName}.npy`;

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
    if (isBuffer) return () => { alive = false; };   // metadata only; never fetch here

    (async () => {
      setStatsBusy(true);
      try {
        const [s, h] = await Promise.all([
          statsOf(value),
          (async () => {
            if (value && typeof value === "object" && !Array.isArray(value) && "values" in (value as object)) {
              const flat = collect((value as { values: unknown }).values);
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
      const info = dtypeInfo(dtypeName);
      const expectedBytes = flatShape(shapeArr) * info.itemsize;

      let payload: Uint8Array;
      const src: unknown = value;
      if (isBuffer && url) {
        // Large payload: the artifact already holds the raw, dtype-correct
        // little-endian bytes (C-order) — wrap it verbatim, no re-encode.
        const ab = await (await fetch(url)).arrayBuffer();
        payload = new Uint8Array(ab);
      } else {
        let flat: number[] = [];
        if (src && typeof src === "object" && !Array.isArray(src) && "values" in (src as object)) {
          flat = collect((src as { values: unknown }).values);
        } else if (Array.isArray(src)) {
          flat = collect(src);
        }
        if (flatShape(shapeArr) > 0 && flat.length !== flatShape(shapeArr)) {
          console.warn(`[tensor] inline count=${flat.length} != shape prod=${flatShape(shapeArr)}; writing anyway`);
        }
        payload = encodeFlat(flat, dtypeName);
      }

      // Sanity: payload byte size should equal elements * itemsize.
      if (expectedBytes > 0 && payload.length !== expectedBytes) {
        console.warn(`[tensor] payload ${payload.length} B != expected ${expectedBytes} B (${dtypeName} ${tupleKey})`);
      }
      saveNpy(payload, dtypeName, shapeArr, npyFileName);
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
            title={`Download .npy (dtype ${dtypeName}, C-order)`}
          >
            {downloading ? "…" : `↓ .npy (${dtypeName})`}
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
