/** Similarity/heatmap view for 2-D numeric values (inline arrays).
 *  Optional axis labels (row/col) + hover readout so a small similarity
 *  matrix reads as "image i × prompt j → value", not just colored cells. */
import { useEffect, useRef, useState } from "react";
import { flattenNumbers, isRef, inlineValues, shapeStr } from "../resolvers";

export function MatrixHeatmap({
  value, title, rowLabels, colLabels,
}: {
  value: unknown;
  title?: string;
  rowLabels?: string[];
  colLabels?: string[];
}) {
  const [mat, setMat] = useState<{ rows: number; cols: number; data: number[] } | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    setMat(null); setNote(null);
    const shape = isRef(value) ? ((value as { shape?: number[] }).shape ?? []) : [];
    let out: { rows: number; cols: number; data: number[] } | null = null;
    let err: string | null = null;
    if (shape.length >= 2) {
      const flat = inlineValues(value as never);
      if (flat && flat.length === shape.reduce((a, b) => a * b, 1)) {
        out = { rows: shape[0], cols: shape[1], data: flat };
      } else {
        err = "artifact: download below and open numerically (heatmap previews inline values only)";
      }
    } else {
      const rows = Array.isArray(value) ? value : null;
      if (rows && rows.every((r) => Array.isArray(r))) {
        const cols = Math.max(...rows.map((r) => r.length));
        const data: number[] = [];
        for (const r of rows) for (let i = 0; i < cols; i++) data.push(Number((r as unknown[])[i]) || 0);
        out = { rows: rows.length, cols, data };
      } else {
        const flat = flattenNumbers(value);
        if (flat && flat.length < 256) {
          // 1-D vector: render 1 x n
          out = { rows: 1, cols: flat.length, data: flat };
        } else {
          err = "not a 2-D value";
        }
      }
    }
    if (out) setMat(out);
    if (err) setNote(err);
  }, [value]);

  if (!mat) {
    return (
      <div>
        {title && <div className="viz-caption">{title}</div>}
        {note && <div className="note">{note}</div>}
        {isRef(value) && (
          <span className="artifact">
            <span className="kind">artifact · {shapeStr((value as { shape?: number[] }).shape)}</span>
            <a href={(value as { url?: string }).url} download>download</a>
          </span>
        )}
      </div>
    );
  }

  const min = Math.min(...mat.data);
  const max = Math.max(...mat.data);
  const rowOk = rowLabels && rowLabels.length === mat.rows && mat.rows <= 200;
  const colOk = colLabels && colLabels.length === mat.cols && mat.cols <= 200;

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      <div style={{ marginBottom: 6, color: "var(--fg-dim)", fontSize: 12 }}>
        {mat.rows} × {mat.cols} · min {min.toFixed(4)} · max {max.toFixed(4)}
      </div>
      <div className="matrixbox">
        <HeatCanvas
          rows={mat.rows} cols={mat.cols} data={mat.data}
          rowLabels={rowOk ? rowLabels! : undefined}
          colLabels={colOk ? colLabels! : undefined}
        />
      </div>
    </div>
  );
}

function trunc(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

/** Viridis — perceptually uniform sequential colormap: dark (low)
 *  → purple → teal → yellow (high).  One shared scale means "bright
 *  = more similar" reads the same on every box's heatmap. */
const VIRIDIS: [number, number, number][] = [
  "#440154", "#46327e", "#365c8d", "#277f8e", "#1fa187", "#4ac16d", "#a0da39", "#fde725",
].map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)] as [number, number, number]);

export function heatColor(t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

export const HEAT_GRADIENT = `linear-gradient(to right, ${VIRIDIS.map(([r, g, b]) => `rgb(${r},${g},${b})`).join(", ")})`;

interface Metrics { x0: number; y0: number; cw: number; ch: number; }

function HeatCanvas({
  rows, cols, data, rowLabels, colLabels,
}: {
  rows: number;
  cols: number;
  data: number[];
  rowLabels?: string[];
  colLabels?: string[];
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const metrics = useRef<Metrics>({ x0: 0, y0: 0, cw: 1, ch: 1 });
  const [ready, setReady] = useState(false);
  const [hover, setHover] = useState<string | null>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    setHover(null);

    const font = "11px system-ui, sans-serif";
    ctx.font = font;

    // cell size (cap total drawn area), plus label gutters
    const cell = Math.max(3, Math.min(34, Math.floor(3_000_000 / (rows * cols || 1))));
    const showRows = !!rowLabels && rowLabels.length === rows;
    const showCols = !!colLabels && colLabels.length === cols && cols <= 40;
    const maxColW = showCols
      ? colLabels!.reduce((m, l) => Math.max(m, ctx.measureText(trunc(l, 28)).width), 0)
      : 0;
    // labels fit above their cells? otherwise draw them rotated (all still labeled)
    const colsFlat = maxColW * 1.15 <= cell * 1.6 && cols <= 12;
    let x0 = 0, y0 = Math.floor(cell / 2) + 4;
    if (showRows) {
      const w = rowLabels!.reduce((m, l) => Math.max(m, ctx.measureText(trunc(l, 24)).width), 0);
      x0 = Math.min(240, Math.ceil(w) + 12);
    }
    if (showCols) {
      y0 = colsFlat ? 28 : Math.min(100, Math.ceil(maxColW) + 14);
    }

    cv.width = Math.max(8, Math.round(x0 + cols * cell));
    cv.height = Math.max(8, Math.round(y0 + rows * cell));
    metrics.current = { x0, y0, cw: cell, ch: cell };

    const min = Math.min(...data);
    const max = Math.max(...data);
    const span = max - min || 1;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const t = (data[i * cols + j] - min) / span; // 0..1
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 1, cell + 1);
      }
    }

    // axis labels
    ctx.fillStyle = "#aab6c4";
    if (showRows) {
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i < rows; i++) {
        ctx.fillText(trunc(rowLabels![i], 24), x0 - 7, y0 + i * cell + cell / 2);
      }
    }
    if (showCols) {
      if (colsFlat) {
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        for (let j = 0; j < cols; j++) {
          ctx.fillText(trunc(colLabels![j], 18), x0 + j * cell + cell / 2, y0 - 6);
        }
      } else {
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        for (let j = 0; j < cols; j++) {
          ctx.save();
          ctx.translate(x0 + j * cell + cell / 2, y0 - 6);
          ctx.rotate(-Math.PI / 2);            // vertical: every column keeps its label
          ctx.fillText(trunc(colLabels![j], 28), 0, 0);
          ctx.restore();
        }
      }
    }
    setReady(true);
  }, [rows, cols, data, rowLabels, colLabels]);

  function onMove(e: React.MouseEvent<HTMLCanvasElement>): void {
    const cv = ref.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const sx = cv.width / rect.width;
    const sy = cv.height / rect.height;
    const px = (e.clientX - rect.left) * sx;
    const py = (e.clientY - rect.top) * sy;
    const { x0, y0, cw, ch } = metrics.current;
    const j = Math.floor((px - x0) / cw);
    const i = Math.floor((py - y0) / ch);
    if (i < 0 || i >= rows || j < 0 || j >= cols) { setHover(null); return; }
    const v = data[i * cols + j];
    const rn = rowLabels && rowLabels.length === rows ? rowLabels[i] : `row ${i + 1}`;
    const cn = colLabels && colLabels.length === cols ? colLabels[j] : `col ${j + 1}`;
    setHover(`${trunc(rn, 32)} × ${trunc(cn, 32)} → ${+v.toPrecision(4)}`);
  }

  return (
    <div>
      <canvas
        ref={ref}
        style={{ maxWidth: "100%", maxHeight: 420, visibility: ready ? "visible" : "hidden", cursor: "crosshair" }}
        onMouseMove={(e) => onMove(e)}
        onMouseLeave={() => setHover(null)}
      />
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-dim)" }}>
        <span>{(+Math.min(...data)).toPrecision(3)}</span>
        <div style={{ flex: "0 1 180px", height: 8, borderRadius: 4, background: HEAT_GRADIENT }} />
        <span>{(+Math.max(...data)).toPrecision(3)}</span>
        <span style={{ color: "var(--fg-dim)" }}>(min → max, normalized)</span>
      </div>
      {hover && (
        <div style={{ marginTop: 4, fontFamily: "var(--mono)", fontSize: 12, color: "var(--fg)" }}>
          {hover}
        </div>
      )}
    </div>
  );
}
