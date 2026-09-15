/** Tensor/array view: dtype/shape/size + stats.
 *
 * Bandwidth rule: inline values (`kind: "array"`) already carry their numbers
 * in the API JSON, so stats + head preview are free. Buffer artifacts
 * (`kind: "buffer"`, e.g. depth maps / world points) show metadata only —
 * the payload is downloaded only if the user clicks "download raw". */
import { useEffect, useState } from "react";
import { isRef, shapeStr, statsOf } from "../resolvers";
import { fmtNum, bytesShort } from "../ui";

export function TensorView({ value, title }: { value: unknown; title?: string }) {
  const [stats, setStats] = useState<{ min: number; max: number; mean: number; n: number } | null>(null);
  const [statsBusy, setStatsBusy] = useState(false);
  const [head, setHead] = useState<number[] | null>(null);
  const [kind, setKind] = useState<string | null>(null);
  const [dtype, setDtype] = useState<string>("");
  const [shape, setShape] = useState<string>("scalar");
  const [size, setSize] = useState<number | null>(null);

  const isBuffer = isRef(value) && (value as { kind?: string }).kind === "buffer";

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

    // Buffer payloads: metadata only — never fetch the raw bytes here.
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

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      <div className="tensorhead">
        <span className="kv"><b>type</b>{kind === "array" ? "inline array" : kind ?? "value"}</span>
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
        {url && <a className="btn small" href={url} download>download raw</a>}
      </div>
      {isBuffer && (
        <div className="viz-note">
          large payload — metadata only; fetch the bytes with "download raw" when needed
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

function walk(v: unknown, out: number[]) {
  if (typeof v === "number" && Number.isFinite(v)) { out.push(v); return; }
  if (Array.isArray(v)) for (const x of v) walk(x, out);
}
