/** Overlay visualizer: base image + typed layers (box / mask / point / flow)
 *  per decoded item.  Coordinates are in the original image's pixel space. */
import { useEffect, useRef, useState } from "react";
import type { LayerDef } from "../api";
import { fetchTyped, inlineValues, isRef, PALETTE } from "../resolvers";

interface OverlayItemProps {
  index: number;
  item: Record<string, unknown>;
  baseUrl: string | null;
  layers: LayerDef[];
}

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
            index={i}
            item={(it && typeof it === "object") ? it as Record<string, unknown> : { value: it }}
            baseUrl={baseImages[i] ?? null}
            layers={layers}
          />
        </div>
      ))}
    </div>
  );
}

function OverlayItem({ index, item, baseUrl, layers }: OverlayItemProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

        // 2) layers
        for (let li = 0; li < layers.length; li++) {
          const layer = layers[li];
          const raw = item[layer.prop];
          if (raw === undefined || raw === null) continue;
          const color = PALETTE[(index + li) % PALETTE.length];
          await drawLayer(ctx, layer, raw, W, H, color);
        }
        if (alive) { setReady(true); setError(null); }
      } catch (e) {
        if (alive) setError(String((e as Error).message || e));
      }
    })();
    return () => { alive = false; };
  }, [item, baseUrl, layers, index]);

  return (
    <div className="overlaybox" ref={wrapRef}>
      <canvas ref={canvasRef} style={{ visibility: ready ? "visible" : "hidden" }} />
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
  raw: unknown,
  W: number,
  H: number,
  color: string,
): Promise<void> {
  const opacity = layer.opacity ?? 1;
  switch (layer.layer) {
    case "box": {
      const rects = await toRects(raw);
      ctx.save();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      for (const r of rects) {
        ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      }
      ctx.restore();
      return;
    }
    case "mask": {
      const mask = await toMaskData(raw);
      if (!mask) return;
      const off = document.createElement("canvas");
      off.width = mask.w; off.height = mask.h;
      const octx = off.getContext("2d");
      if (!octx) return;
      const data = octx.createImageData(mask.w, mask.h);
      const [r, g, b] = hexRgb(color);
      for (let i = 0; i < mask.data.length; i++) {
        const v = mask.data[i];
        data.data[i * 4 + 0] = r;
        data.data[i * 4 + 1] = g;
        data.data[i * 4 + 2] = b;
        data.data[i * 4 + 3] = Math.round(Math.min(1, Math.max(0, v)) * 255 * opacity * 0.85);
      }
      octx.putImageData(data, 0, 0);
      ctx.drawImage(off, 0, 0, off.width, off.height, 0, 0, W, H);
      return;
    }
    case "point":
    case "flow": {
      const pts = await toPoints(raw);
      ctx.save();
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
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

async function toMaskData(raw: unknown): Promise<{ w: number; h: number; data: number[] } | null> {
  if (Array.isArray(raw)) {
    // inline: [h][w] number matrix, possibly wrapped in a single-element list
    let arr: unknown = raw;
    while (Array.isArray(arr) && (arr as unknown[]).length === 1 && Array.isArray((arr as unknown[])[0])) {
      arr = (arr as unknown[])[0];
    }
    if (
      Array.isArray(arr) &&
      (arr as unknown[]).length > 0 &&
      (arr as unknown[]).every((r) => Array.isArray(r) && (r as unknown[]).length > 0 &&
        (r as unknown[]).every((x) => typeof x === "number"))
    ) {
      const rows = arr as number[][];
      const w = Math.max(...rows.map((r) => r.length));
      const h = rows.length;
      const data: number[] = [];
      for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) data.push(Number(rows[i][j]) || 0);
      return { w, h, data };
    }
    return null;
  }
  if (isRef(raw)) {
    const shape = ((raw as { shape?: number[] }).shape ?? []) as number[];
    const t = await fetchTyped(raw as never);
    if (!t) return null;
    const n = t.data.length;
    if (shape.length >= 2) {
      const w = shape[shape.length - 1];
      const h = shape[shape.length - 2];
      if (n >= w * h) {
        const data: number[] = [];
        for (let i = 0; i < w * h; i++) data.push(Number(t.data[i]));
        return { w, h, data };
      }
    }
    // 1-D fallback: square-ish guess
    const w = Math.round(Math.sqrt(n));
    const h = Math.ceil(n / w);
    const data: number[] = [];
    for (let i = 0; i < n; i++) data.push(Number(t.data[i]));
    return { w, h, data };
  }
  return null;
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
