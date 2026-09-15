/** Overlay visualizer: base image + typed layers (box / mask / point / flow)
 *  per decoded item.  Coordinates are in the original image's pixel space.
 *
 *  Mask layers are explicit by construction:
 *   - each mask i gets its own cycle color (PALETTE[i]), and box layer
 *     rectangles are colored the same way, so box i ↔ mask i pair up;
 *   - the fill is strong (layer.opacity, default 0.7) plus a light outline
 *     drawn along the mask boundary so regions read clearly on any photo;
 *   - a numbered legend under the canvas says what each colored region
 *     represents: label (text_labels / labels, …) and score (scores /
 *     mask_scores, …) from the item, if the box reported them. */
import { useEffect, useRef, useState } from "react";
import type { LayerDef } from "../api";
import { fetchTyped, inlineValues, isRef, PALETTE } from "../resolvers";

interface OverlayItemProps {
  item: Record<string, unknown>;
  baseUrl: string | null;
  layers: LayerDef[];
}

interface LegendEntry {
  color: string;
  id: number;
  label: string;
  score: number | null;
}

type MaskData = { w: number; h: number; data: number[] };

export function OverlayViz({
  items, baseImages, layers,
}: {
  items: unknown[];
  baseImages: (string | null)[];
  layers: LayerDef[];
}) {
  if (!items.length) return <div className="note">no items to overlay</div>;
  return (
    <div>
      {items.map((it, i) => (
        <div key={i} className="overlay-per">
          {(items.length > 1 || i > 0) && <div className="cap viz-caption">item {i}</div>}
          <OverlayItem
            item={(it && typeof it === "object") ? it as Record<string, unknown> : { value: it }}
            baseUrl={baseImages[i] ?? null}
            layers={layers}
          />
        </div>
      ))}
    </div>
  );
}

function OverlayItem({ item, baseUrl, layers }: OverlayItemProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [legend, setLegend] = useState<LegendEntry[]>([]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const cv = canvasRef.current;
        if (!cv) return;
        const ctx = cv.getContext("2d");
        if (!ctx) return;

        // 1) base image (drives the canvas coordinate space)
        let W = 640, H = 480;
        if (baseUrl) {
          const img = await loadImage(baseUrl);
          if (img) { W = img.naturalWidth; H = img.naturalHeight; }
        }
        cv.width = W;
        cv.height = H;
        if (baseUrl) {
          const img = await loadImage(baseUrl);
          if (img) ctx.drawImage(img, 0, 0, W, H);
        } else {
          ctx.fillStyle = "#10151c";
          ctx.fillRect(0, 0, W, H);
        }
        if (!alive) return;

        // 2) layers (mask layers push into the shared legend)
        const legend: LegendEntry[] = [];
        for (const layer of layers) {
          const raw = item[layer.prop];
          if (raw === undefined || raw === null) continue;
          await drawLayer(ctx, layer, item, raw, W, H, legend);
        }
        if (alive) { setLegend(legend); setReady(true); setError(null); }
      } catch (e) {
        if (alive) setError(String((e as Error).message || e));
      }
    })();
    return () => { alive = false; };
  }, [item, baseUrl, layers]);

  return (
    <div className="overlaybox">
      <canvas ref={canvasRef} style={{ visibility: ready ? "visible" : "hidden" }} />
      {legend.length > 0 && (
        <div className="legend mask-legend">
          {legend.map((e, k) => (
            <span key={k} className="row" title={`${e.label} ${e.score != null ? e.score.toFixed(2) : ""}`}>
              <i className="sw" style={{ background: e.color }} />
              {` ${e.id + 1} · ${e.label}${e.score != null ? ` (${e.score.toFixed(2)})` : ""}`}
            </span>
          ))}
        </div>
      )}
      {error && <div className="note" style={{ color: "var(--err)" }}>overlay failed: {error}</div>}
    </div>
  );
}

function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

async function drawLayer(
  ctx: CanvasRenderingContext2D,
  layer: LayerDef,
  item: Record<string, unknown>,
  raw: unknown,
  W: number,
  H: number,
  legend: LegendEntry[],
): Promise<void> {
  const opacity = layer.opacity ?? 1;
  switch (layer.layer) {
    case "box": {
      // one color per rectangle, matching the mask color cycle below, so
      // box i visually belongs to mask i
      const rects = await toRects(raw);
      ctx.save();
      ctx.lineWidth = 2.5;
      for (let i = 0; i < rects.length; i++) {
        const r = rects[i];
        ctx.strokeStyle = PALETTE[i % PALETTE.length];
        ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      }
      ctx.restore();
      return;
    }
    case "mask": {
      const masks = await toMaskList(raw);
      if (!masks.length) return;
      const labels = labelsPerMask(item);
      const scores = await scoresPerMask(item);
      for (let mi = 0; mi < masks.length; mi++) {
        const color = PALETTE[mi % PALETTE.length];
        const [r, g, b] = hexRgb(color);
        drawMaskFill(ctx, masks[mi], W, H, r, g, b, opacity);
        drawMaskOutline(ctx, masks[mi], W, H, r, g, b);
        const s = scores ? scores[mi] : undefined;
        legend.push({
          color,
          id: mi,
          label: labels && labels[mi] ? labels[mi] : `mask ${mi + 1}`,
          score: typeof s === "number" && Number.isFinite(s) ? s : null,
        });
      }
      return;
    }
    case "point":
    case "flow": {
      const pts = await toPoints(raw);
      const [r, g, b] = hexRgb(PALETTE[0]);
      ctx.save();
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      for (const p of pts) {
        if (layer.layer === "flow" && p.length >= 4) {
          ctx.beginPath();
          ctx.moveTo(p[0], p[1]);
          ctx.lineTo(p[0] + p[2], p[1] + p[3]);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(p[0], p[1], 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
      return;
    }
    default:
      return;
  }
}

/** Opaque-ish region fill, full resolution. */
function drawMaskFill(
  ctx: CanvasRenderingContext2D,
  mask: MaskData,
  W: number,
  H: number,
  r: number,
  g: number,
  b: number,
  opacity: number,
): void {
  const off = document.createElement("canvas");
  off.width = mask.w; off.height = mask.h;
  const octx = off.getContext("2d");
  if (!octx) return;
  const data = octx.createImageData(mask.w, mask.h);
  for (let i = 0; i < mask.data.length; i++) {
    const v = Math.min(1, Math.max(0, mask.data[i] || 0));
    data.data[i * 4 + 0] = r;
    data.data[i * 4 + 1] = g;
    data.data[i * 4 + 2] = b;
    data.data[i * 4 + 3] = Math.round(v * 255 * Math.min(1, opacity));
  }
  octx.putImageData(data, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, off.width, off.height, 0, 0, W, H);
}

/** Light boundary outline along the mask edge (cheap: computed at reduced
 *  resolution, then scaled up — the edge is 1 px-ish either way). */
function drawMaskOutline(
  ctx: CanvasRenderingContext2D,
  mask: MaskData,
  W: number,
  H: number,
  r: number,
  g: number,
  b: number,
): void {
  const maxSide = 320;
  const scale = Math.min(1, maxSide / Math.max(mask.w, mask.h));
  const w = Math.max(4, Math.round(mask.w * scale));
  const h = Math.max(4, Math.round(mask.h * scale));
  const on = (x: number, y: number): boolean => {
    if (x < 0 || y < 0 || x >= mask.w || y >= mask.h) return false;
    return (mask.data[y * mask.w + x] || 0) >= 0.5;
  };
  // lightened version of the mask color: readable over both dark and
  // bright photo regions the fill sits on
  const lr = Math.min(255, Math.round(r + (255 - r) * 0.75));
  const lg = Math.min(255, Math.round(g + (255 - g) * 0.75));
  const lb = Math.min(255, Math.round(b + (255 - b) * 0.75));

  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const octx = off.getContext("2d");
  if (!octx) return;
  const data = octx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = Math.min(mask.w - 1, Math.round(x / scale));
      const sy = Math.min(mask.h - 1, Math.round(y / scale));
      if (!on(sx, sy)) continue;
      if (on(sx - 1, sy) && on(sx + 1, sy) && on(sx, sy - 1) && on(sx, sy + 1)) continue;
      data.data[(y * w + x) * 4 + 0] = lr;
      data.data[(y * w + x) * 4 + 1] = lg;
      data.data[(y * w + x) * 4 + 2] = lb;
      data.data[(y * w + x) * 4 + 3] = 235;
    }
  }
  octx.putImageData(data, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
}

// ---------------------------------------------------------------------------
// per-mask labels / scores (item-level lists, parallel to the mask list)
// ---------------------------------------------------------------------------

const LABEL_KEYS = ["text_labels", "labels", "text_prompt", "prompts", "text"];
const SCORE_KEYS = ["scores", "mask_scores", "label_scores"];

function labelsPerMask(item: Record<string, unknown>): string[] | null {
  for (const key of LABEL_KEYS) {
    const s = strList(item[key]);
    if (s && s.length) return s;
  }
  return null;
}

async function scoresPerMask(item: Record<string, unknown>): Promise<number[] | null> {
  for (const key of SCORE_KEYS) {
    const v = item[key];
    if (v === undefined || v === null) continue;
    if (typeof v === "number" && Number.isFinite(v)) return [v];
    if (Array.isArray(v)) {
      // element may be a nested per-phrase score list -> take the max
      const out = v.map((el) => {
        if (typeof el === "number" && Number.isFinite(el)) return el;
        if (Array.isArray(el)) {
          const nums = (el as unknown[]).filter((x) => typeof x === "number" && Number.isFinite(x)) as number[];
          return nums.length ? Math.max(...nums) : NaN;
        }
        return NaN;
      });
      if (out.length) return out;
      continue;
    }
    if (isRef(v)) {
      // serialized tensor: inline values, or fetch the buffer artifact
      const t = await fetchTyped(v as never);
      if (t && t.data.length) {
        const out: number[] = [];
        for (let i = 0; i < t.data.length; i++) out.push(Number(t.data[i]));
        return out;
      }
    }
  }
  return null;
}

function strList(v: unknown): string[] | null {
  const list = Array.isArray(v) ? v : (typeof v === "string" ? [v] : null);
  if (!list) return null;
  const out: string[] = [];
  for (const el of list) {
    if (typeof el === "string" && el.trim()) { out.push(el.trim()); continue; }
    if (typeof el === "number") { out.push(String(el)); continue; }
    if (Array.isArray(el)) {
      const parts = (el as unknown[])
        .filter((x) => typeof x === "string" && String(x).trim())
        .map((x) => String(x).trim());
      if (parts.length) { out.push(parts.join(", ")); continue; }
    }
    out.push(""); // opaque entry (ref / object) — legend falls back to "mask i"
  }
  return out;
}

// ---------------------------------------------------------------------------
// geometry helpers
// ---------------------------------------------------------------------------

async function toRects(raw: unknown): Promise<number[][]> {
  // shape (n,4) or ((1,n,4)) flat, or list of [x1,y1,x2,y2]
  if (Array.isArray(raw)) {
    const flat: number[] = [];
    for (const x of raw) {
      if (Array.isArray(x)) {
        const inner = Array.isArray(x[0]) ? (x as unknown[][])[0] : x;
        for (const y of (inner as unknown[])) if (typeof y === "number") flat.push(y);
      } else if (typeof x === "number") flat.push(x);
    }
    if (flat.length && flat.length % 4 === 0) {
      return Array.from({ length: flat.length / 4 }, (_, i) =>
        [flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2], flat[i * 4 + 3]]);
    }
    return [];
  }
  if (isRef(raw)) {
    const t = await fetchTyped(raw as never);
    if (t) {
      const n = Math.floor(t.data.length / 4);
      return Array.from({ length: n }, (_, i) =>
        [t.data[i * 4], t.data[i * 4 + 1], t.data[i * 4 + 2], t.data[i * 4 + 3]]);
    }
    const inline = inlineValues(raw as never);
    if (inline && inline.length % 4 === 0) {
      return Array.from({ length: inline.length / 4 }, (_, i) =>
        [inline[i * 4], inline[i * 4 + 1], inline[i * 4 + 2], inline[i * 4 + 3]]);
    }
  }
  return [];
}

/** Masks from a raw field value: a flat inline matrix (single mask), a list
 *  of matrices/refs (one entry per mask), or one ref with shape
 *  (n,h,w) / (h,w) / (n,). */
async function toMaskList(raw: unknown): Promise<MaskData[]> {
  if (Array.isArray(raw)) {
    // single inline mask matrix?  [h][w] numbers
    const isMatrix = raw.length > 0 &&
      raw.every((x) => Array.isArray(x) && (x as unknown[]).length > 0 &&
        (x as unknown[]).every((y) => typeof y === "number"));
    if (isMatrix) {
      const m = inlineMatrixToMask(raw);
      return m ? [m] : [];
    }
    const out: MaskData[] = [];
    for (const el of raw) {
      if (Array.isArray(el)) {
        const m = inlineMatrixToMask(el);
        if (m) out.push(m);
      } else if (isRef(el)) {
        out.push(...await refToMasks(el));
      }
    }
    return out;
  }
  if (isRef(raw)) {
    return refToMasks(raw);
  }
  return [];
}

function inlineMatrixToMask(raw: unknown): MaskData | null {
  if (!Array.isArray(raw) || !raw.length) return null;
  if (!raw.every((r) => Array.isArray(r) && (r as unknown[]).length > 0 &&
    (r as unknown[]).every((x) => typeof x === "number"))) return null;
  const rows = raw as number[][];
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const data: number[] = [];
  for (let i = 0; i < h; i++) for (let j = 0; j < (rows[i]?.length ?? 0); j++) data.push(Number(rows[i][j]) || 0);
  return { w, h, data };
}

async function refToMasks(raw: unknown): Promise<MaskData[]> {
  const t = await fetchTyped(raw as never);
  if (!t || t.data.length === 0) {
    const inline = inlineValues(raw as never);
    if (!inline || inline.length === 0) return [];
    const n = inline.length;
    const w = Math.max(1, Math.round(Math.sqrt(n)));
    const h = Math.ceil(n / w);
    return [{ w, h, data: inline.slice(0, w * h) }];
  }
  const flat: number[] = [];
  const push = (count: number) => { for (let i = 0; i < count; i++) flat.push(Number(t.data[i])); };
  const shape = t.shape;
  if (shape.length >= 3) {
    const n = Math.max(1, shape[0]), h = Math.max(1, shape[1]), w = Math.max(1, shape[2]);
    const total = Math.min(t.data.length, n * h * w);
    push(total);
    const out: MaskData[] = [];
    for (let i = 0; i < n; i++) out.push({ w, h, data: flat.slice(i * w * h, Math.min((i + 1) * w * h, flat.length)) });
    return out;
  }
  if (shape.length === 2) {
    const h = Math.max(1, shape[0]), w = Math.max(1, shape[1]);
    push(Math.min(t.data.length, h * w));
    return [{ w, h, data: flat }];
  }
  const n = t.data.length;
  const w = Math.max(1, Math.round(Math.sqrt(n)));
  const h = Math.ceil(n / w);
  push(n);
  return [{ w, h, data: flat.slice(0, w * h) }];
}

async function toPoints(raw: unknown): Promise<number[][]> {
  // (n,2) flat | list of [x,y] | (n,4) flow
  const flat: number[] = [];
  if (Array.isArray(raw)) {
    for (const x of raw) {
      if (Array.isArray(x)) for (const y of x) if (typeof y === "number") flat.push(y);
      else if (typeof x === "number") flat.push(x);
    }
  } else if (isRef(raw)) {
    const t = await fetchTyped(raw as never);
    if (t) for (let i = 0; i < t.data.length; i++) flat.push(Number(t.data[i]));
  }
  const dim = flat.length && shapeGuess(raw) === 4 ? 4 : 2;
  const out: number[][] = [];
  for (let i = 0; i + dim <= flat.length; i += dim) out.push(flat.slice(i, i + dim));
  return out;
}

function shapeGuess(raw: unknown): number {
  if (isRef(raw)) {
    const shape = ((raw as { shape?: number[] }).shape ?? []) as number[];
    return shape[shape.length - 1] ?? 2;
  }
  if (Array.isArray(raw)) {
    const first = raw[0];
    if (Array.isArray(first)) return first.length;
  }
  return 2;
}

function hexRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}
