/** field_map visualizer — render a *numeric map field* of per-item result
 *  dicts as an image, box-agnostic:
 *
 *   - (H, W)    -> viridis heat image + low/high legend (same palette and
 *     "bright = more" semantics as the matrix heatmaps)
 *   - (H, W, 3) -> RGB image (uint8 passes through; float channels are
 *     min/max normalized per channel) — e.g. normals or point maps as RGB
 *   - otherwise -> shape metadata + download link
 *
 *  Values may be inline arrays or buffer artifacts (fetched only because
 *  this visualizer must render them).  Non-numeric cells degrade, never
 *  break.  Nothing here knows a box by name: the def's `params.prop` picks
 *  the field per item.
 */
import { useEffect, useRef, useState } from "react";
import { itemNumerics, shapeStr } from "../resolvers";
import { HEAT_GRADIENT, heatColor } from "./MatrixHeatmap";

const MAX_DRAW = 1024;        // render resolution cap (offscreen stays full res)

interface ItemMap {
  data: ArrayLike<number>;
  shape: number[];
  url: string | null;
}

export function FieldMap({ value, prop, title }: {
  value: unknown;
  prop?: string;
  title?: string;
}) {
  const [maps, setMaps] = useState<ItemMap[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setMaps(null);
    setErr(null);
    void (async () => {
      const m = await itemNumerics(value, prop);
      if (!alive) return;
      if (m.length === 0) setErr(`no numeric ${prop ? `“${prop}”` : "map"} values in the response`);
      else setMaps(m);
    })();
    return () => { alive = false; };
  }, [value, prop]);

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      {err && <div className="note">{err}</div>}
      {maps === null && !err && <span className="spinnerbox"><span className="spinner" /></span>}
      {maps && maps.map((m, i) => (
        <div key={i} className="overlay-per">
          {(maps.length > 1 || i > 0) && <div className="cap viz-caption">img {i + 1}</div>}
          <MapCanvas map={m} />
        </div>
      ))}
    </div>
  );
}

function MapCanvas({ map }: { map: ItemMap }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const mode: "heat" | "rgb" | "meta" =
    map.shape.length === 2 ? "heat"
      : (map.shape.length === 3 && map.shape[2] === 3) ? "rgb"
      : "meta";

  useEffect(() => {
    let alive = true;
    setReady(false);
    setNote(null);
    const notes: string[] = [shapeStr(map.shape)];
    let drawn = false;

    if (mode === "heat") {
      const [h, w] = [map.shape[0], map.shape[1]];
      if (!h || !w || map.data.length < h * w) {
        if (alive) setNote("shape/data mismatch");
        return;
      }
      let mn = Infinity, mx = -Infinity, invalid = 0;
      for (let i = 0; i < h * w; i++) {
        const x = Number(map.data[i]);
        if (!Number.isFinite(x)) { invalid++; continue; }   // NaN/Inf regions exist in real maps
        if (x < mn) mn = x;
        if (x > mx) mx = x;
      }
      if (!Number.isFinite(mn)) {
        if (alive) setNote(notes.concat("no finite values — nothing to render").join(" · "));
        return;
      }
      const span = mx - mn || 1;
      const off = document.createElement("canvas");
      off.width = w; off.height = h;
      const octx = off.getContext("2d");
      if (!octx) return;
      const img = octx.createImageData(w, h);
      for (let i = 0; i < h * w; i++) {
        const x = Number(map.data[i]);
        const [r, g, b] = Number.isFinite(x) ? heatRgb((x - mn) / span) : heatRgb(NaN);
        img.data[i * 4 + 0] = r;
        img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      draw(ref.current, off);
      drawn = true;
      if (alive) setNote(notes.concat(
        `${w} × ${h} · min ${mn.toPrecision(3)} · max ${mx.toPrecision(3)}`,
        invalid ? `${invalid} invalid (dark) cells` : "",
      ).filter(Boolean).join(" · "));
    } else if (mode === "rgb") {
      const [h, w] = [map.shape[0], map.shape[1]];
      if (!h || !w || map.data.length < h * w * 3) {
        if (alive) setNote("shape/data mismatch");
        return;
      }
      const isU8 = (map.data as Uint8Array).constructor === Uint8Array;
      // (H, W, 3) is INTERLEAVED: pixel i, channel c is at i*3+c (not planar)
      const stats: { mn: number; mx: number }[] = [];
      for (let c = 0; c < 3; c++) {
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < h * w; i++) {
          const x = Number(map.data[i * 3 + c]);
          if (!Number.isFinite(x)) continue;
          if (x < mn) mn = x;
          if (x > mx) mx = x;
        }
        stats.push({ mn, mx });
      }
      // unit-vector data (normals: each channel in [-1, 1]) uses the standard
      // symmetric (v+1)/2 RGB encoding; anything else is min/max per channel
      const symmetric = !isU8 && stats.every((s) =>
        Number.isFinite(s.mn) && s.mn >= -1.0001 && s.mx <= 1.0001);
      const off = document.createElement("canvas");
      off.width = w; off.height = h;
      const octx = off.getContext("2d");
      if (!octx) return;
      const img = octx.createImageData(w, h);
      for (let i = 0; i < h * w; i++) {
        for (let c = 0; c < 3; c++) {
          const x = Number(map.data[i * 3 + c]);
          if (!Number.isFinite(x)) { img.data[i * 4 + c] = 0; continue; }
          let t: number;
          if (isU8) {
            t = Math.max(0, Math.min(1, x / 255));
          } else if (symmetric) {
            t = (x + 1) / 2;
          } else {
            const { mn, mx } = stats[c];
            t = (x - mn) / ((mx - mn) || 1);
          }
          img.data[i * 4 + c] = Math.round(255 * Math.max(0, Math.min(1, t)));
        }
        img.data[i * 4 + 3] = 255;             // ImageData starts fully transparent
      }
      octx.putImageData(img, 0, 0);
      draw(ref.current, off);
      drawn = true;
      if (isU8) {
        // nothing extra to note
      } else if (symmetric) {
        notes.push("unit vectors — each channel rendered as (v+1)/2");
      } else {
        stats.forEach((s, c) => notes.push(`ch${c + 1} ${s.mn.toPrecision(2)}…${s.mx.toPrecision(2)}`));
        notes.push("float channels min/max normalized");
      }
    } else {
      notes.push("not an H×W (heat) or H×W×3 (RGB) map — metadata only");
    }

    if (alive) { setReady(drawn); setNote(notes.join(" · ")); }
  }, [map, mode]);

  return (
    <div>
      <canvas
        ref={(el) => { ref.current = el; }}
        style={{ maxWidth: "100%", display: "block", visibility: ready ? "visible" : "hidden" }}
      />
      {note && <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>{note}</div>}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-dim)" }}>
        {mode === "heat" && (
          <>
            <span>low</span>
            <div style={{ flex: "0 1 180px", height: 8, borderRadius: 4, background: HEAT_GRADIENT }} />
            <span>high</span>
            <span style={{ color: "var(--fg-dim)" }}>(viridis, min → max normalized)</span>
          </>
        )}
        {mode === "rgb" && <span>RGB — one channel per value axis</span>}
        {map.url && <a href={map.url} download>download raw</a>}
      </div>
      {!ready && !note && <span className="spinnerbox"><span className="spinner" /></span>}
    </div>
  );
}

function draw(target: HTMLCanvasElement | null, off: HTMLCanvasElement): void {
  if (!target) return;
  const w = Math.min(MAX_DRAW, off.width);
  const h = Math.min(MAX_DRAW, off.height);
  target.width = w;
  target.height = h;
  const ctx = target.getContext("2d");
  if (ctx) ctx.drawImage(off, 0, 0, w, h);
}

/** heatColor() returns an `rgb()` string — parse it so ImageData can use
 *  ints (reusing the shared palette keeps every heatmap consistent). */
function heatRgb(t: number): [number, number, number] {
  const m = heatColor(t).match(/\d+/g);
  return m ? [Number(m[0]), Number(m[1]), Number(m[2])] : [0, 0, 0];
}
