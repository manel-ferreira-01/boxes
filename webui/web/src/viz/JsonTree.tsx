/** Recursive JSON tree that renders serialized refs (file/buffer/array)
 *  as labeled chips with downloads instead of exploding into raw JSON. */
import { useMemo, useState } from "react";
import { isRef, shapeStr } from "../resolvers";
import { bytesShort } from "../ui";

export function valueKind(v: unknown): string | null {
  if (!isRef(v)) return null;
  return v.kind;
}

export function RenderValue({ v, name, depth = 0 }: { v: unknown; name?: string; depth?: number }) {
  // refs -> chip
  if (isRef(v)) {
    const dtype = (v as { dtype?: string }).dtype;
    const shape = (v as { shape?: number[] }).shape;
    const size = (v as { size?: number }).size;
    const mime = (v as { mime?: string }).mime;
    const url = (v as { url?: string }).url;
    return (
      <span className="artifact" title={v.kind}>
        <span className="kind">{v.kind}{dtype ? ` · ${dtype}` : ""}{shape ? ` · ${shapeStr(shape)}` : ""}{mime ? ` · ${mime}` : ""}</span>
        {typeof size === "number" && <span>{bytesShort(size)}</span>}
        {url && <a href={url} target="_blank" rel="noreferrer">open</a>}
        {url && <a href={url} download={name ? `${name}` : undefined}>download</a>}
      </span>
    );
  }
  return <InlineJson v={v} depth={depth} />;
}

export function InlineJson({ v, depth = 0 }: { v: unknown; depth?: number }) {
  if (v === null || v === undefined) return <span className="b">null</span>;
  const t = typeof v;
  if (t === "string") return <span className="s">"{t === "string" ? (v as string).slice(0, 200) : ""}"</span>;
  if (t === "number" || t === "boolean") return <span className="n">{String(v)}</span>;
  if (Array.isArray(v)) {
    if (v.every((x) => typeof x !== "object" || x === null)) {
      return <span className="n">[{v.map(x => String(x).slice(0, 24)).join(", ")}]</span>;
    }
    return (
      <>
        <span className="b">[</span>{v.map((x, i) => (
          <span key={i} className="row"><RenderValue v={x} name={String(i)} depth={depth + 1} /></span>
        ))}<span className="b">]</span>
      </>
    );
  }
  if (t === "object") {
    return (
      <>
        <span className="b">{"{"}</span>
        {Object.entries(v as Record<string, unknown>).map(([k, x]) => (
          <span key={k} className="row">
            <span className="k">{k}</span>: <RenderValue v={x} name={k} depth={depth + 1} />
          </span>
        ))}
        <span className="b">{"}"}</span>
      </>
    );
  }
  return <span className="b">?</span>;
}

export function JsonTree({ data }: { data: unknown }) {
  const [depth, setDepth] = useState(2);
  return (
    <div className="jtree">
      <div style={{ marginBottom: 6 }}>
        <button className="btn small" onClick={() => setDepth((d) => Math.max(1, d - 1))}>shallow</button>{" "}
        <button className="btn small" onClick={() => setDepth((d) => Math.min(9, d + 1))}>deep</button>
      </div>
      <TreeNode v={data} limit={depth} d={0} />
    </div>
  );
}

function TreeNode({ v, limit, d }: { v: unknown; limit: number; d: number }) {
  const collapsible = (v !== null && typeof v === "object") && d < limit;
  const [open, setOpen] = useState(false);
  const len = useMemo(
    () => (Array.isArray(v) ? v.length : typeof v === "object" && v ? Object.keys(v as object).length : null),
    [v],
  );
  if (!collapsible) return <span><RenderValue v={v} /></span>;
  if (open) return <RenderValue v={v} depth={d} />;
  const summary =
    Array.isArray(v) ? `array[${len}]` : `object{${len}}`;
  return (
    <span className="collapserow" onClick={() => setOpen(true)} title="expand">
      <span className="b">{summary} ▸</span>
    </span>
  );
}
