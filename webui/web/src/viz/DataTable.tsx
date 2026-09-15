/** Data table for list-of-objects or 2-D numeric values. */
import { useMemo } from "react";
import { flattenNumbers, isRef, inlineValues, shapeStr } from "../resolvers";

export function DataTable({ value, title }: { value: unknown; title?: string }) {
  const model = useMemo(() => toModel(value), [value]);
  if (!model) return <Fallback value={value} />;
  const rows = model.rows.slice(0, 80);
  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>{model.cols.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>{r.map((c, j) => <td key={j}>{cell(c)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      {model.rows.length > 80 && <div className="note">showing 80 of {model.rows.length} rows</div>}
    </div>
  );
}

function cell(c: unknown): string {
  if (c === null) return "null";
  if (typeof c === "number") return String(Number(c.toPrecision(5)));
  if (typeof c === "object") return JSON.stringify(c).slice(0, 60);
  return String(c).slice(0, 120);
}

function toModel(value: unknown): { cols: string[]; rows: unknown[][] } | null {
  // array of objects -> columns
  if (Array.isArray(value) && value.length && value.every((x) => x && typeof x === "object" && !Array.isArray(x))) {
    const set = new Set<string>();
    for (const x of value) for (const k of Object.keys(x as object)) set.add(k);
    const cols = Array.from(set).slice(0, 10);
    return {
      cols,
      rows: value.map((x) => {
        const o = x as Record<string, unknown>;
        const r = cols.map((c) => o[c]);
        if (set.size > 10) r.push(`…${set.size - 10} more keys`);
        return r;
      }),
    };
  }
  // 2-D number matrix
  if (Array.isArray(value) && value.length && value.every((r) => Array.isArray(r) && r.every((x) => typeof x === "number"))) {
    const width = Math.max(...value.map((r) => r.length));
    return {
      cols: Array.from({ length: width }, (_, i) => `c${i}`),
      rows: value.map((r) => Array.from({ length: width }, (_, j) => (r[j] ?? null))),
    };
  }
  // serialized array ref
  if (isRef(value)) {
    const flat = inlineValues(value as never);
    const shape = ((value as { shape?: unknown }).shape ?? []) as number[];
    if (flat && flat.length) {
      if (shape.length >= 2) {
        const [n, c] = [shape[0], shape.reduce((a, b) => a * b)];
        const per = Math.max(1, Math.round(c / n));
        return {
          cols: Array.from({ length: Math.min(per, 12) }, (_, i) => `c${i}`),
          rows: Array.from({ length: Math.min(n, 200) }, (_, i) =>
            flat.slice(i * per, Math.min(flat.length, (i + 1) * per)).map((x, j) => (j < 12 ? x : null))),
        };
      }
      return { cols: ["value"], rows: flat.slice(0, 200).map((x) => [x]) };
    }
    return null;
  }
  // list of scalars
  const flat = flattenNumbers(value);
  if (flat) return { cols: ["value"], rows: flat.slice(0, 200).map((x) => [x]) };
  return null;
}

function Fallback({ value }: { value: unknown }) {
  if (value === undefined || value === null) {
    // field absent from this response (e.g. a reset/reply without results)
    return <div className="note">no value in this response</div>;
  }
  if (isRef(value)) {
    return (
      <div>
        <span className="artifact">
          <span className="kind">artifact · {shapeStr((value as { shape?: unknown }).shape)}</span>
          {(value as { url?: string }).url && (
            <a href={(value as { url?: string }).url} download>download</a>
          )}
        </span>
        <div className="note">too structured for a table preview — download or view JSON below</div>
      </div>
    );
  }
  return <pre className="json">{JSON.stringify(value, null, 2).slice(0, 4000)}</pre>;
}
