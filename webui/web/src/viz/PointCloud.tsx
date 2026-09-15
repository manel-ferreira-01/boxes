/** points visualizer — per-item 3D point clouds, orbit/zoom.
 *
 *  The easy way: a point cloud *is* typed arrays, so there is no 3D file
 *  format to encode at all.  Positions + colors go straight into a
 *  `THREE.Points` object (one BufferGeometry + a PointsMaterial) — no GLB
 *  writer, no blobs, no binary layout to debug.  (GLB/glTF is only needed
 *  when you want a portable model artifact, e.g. the vggt scene; for
 *  rendering points it is pure overhead.)
 *
 *  One viewer per input image, same per-item cards as field_map.
 *
 *  Position source (the def picks via `params` — nothing here knows a box):
 *   depth + intrinsics — reproject every pixel of the depth map into 3D
 *     with a pinhole back-projection at pixel centers:
 *         X = z · ((u + 0.5)/W − cx) / fx
 *         Y = z · ((v + 0.5)/H − cy) / fy
 *         Z = z
 *     Default `projection: normalized` is the MoGe/MoGE family convention
 *     (fx, cy in width fractions, i.e. the standard pinhole camera with the
 *     image plane scaled to [0, 1)²); `projection: pixel` uses intrinsics in
 *     image-pixel units instead.  When the result def also names a base-image
 *     input (`rd.base`), every point is colored with the input image's RGB
 *     at its pixel — a photo-true 3D scan.
 *
 *   points — fallback: (…, 3) positions straight from the box (no depth
 *     declared), colored by their xyz range (min/max per channel).
 *
 *  Non-finite or ≤ 0 depths are dropped; an optional `mask` prop (same grid)
 *  keeps only cells > 0.5. */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SerValue } from "../api";
import { fetchTyped } from "../resolvers";

interface Cloud {
  positions: Float32Array;   // M × 3, camera (OpenCV) coordinates
  colors: Float32Array;      // M × 3, 0..1
  center: [number, number, number];
  maxDim: number;
  note: string;
}

interface Cfg {
  depth?: string;
  intrinsics?: string;
  projection?: string;   // "normalized" (default) | "pixel"
  mask?: string;
  points?: string;
  baseUrl: string | null;
}

type Item = Cloud | { error: string };

export function PointCloud({
  value, depth, intrinsics, projection, mask, points, baseImages, title,
}: {
  value: unknown;
  depth?: string;
  intrinsics?: string;
  projection?: string;
  mask?: string;
  points?: string;
  baseImages?: string[];
  title?: string;
}) {
  const [items, setItems] = useState<Item[] | null>(null);

  useEffect(() => {
    let alive = true;
    setItems(null);
    const list = Array.isArray(value) ? value : [value];
    void (async () => {
      const out: Item[] = [];
      for (let i = 0; i < list.length; i++) {
        try {
          out.push(await buildItem(list[i], {
            depth, intrinsics, projection, mask, points,
            baseUrl: baseImages && baseImages.length > i ? baseImages[i] : null,
          }));
        } catch (e) {
          out.push({ error: String((e as Error).message || e) });
        }
      }
      if (alive) setItems(out);
    })();
    return () => { alive = false; };
  }, [value, depth, intrinsics, projection, mask, points, baseImages]);

  if (!items) {
    return (
      <div>
        {title && <div className="viz-caption">{title}</div>}
        <span className="spinnerbox"><span className="spinner" /></span>
      </div>
    );
  }

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      {items.map((it, i) =>
        "error" in it ? (
          <div key={i} className="overlay-per">
            {(items.length > 1 || i > 0) && <div className="cap viz-caption">img {i + 1}</div>}
            <div className="note">unavailable: {it.error}</div>
          </div>
        ) : (
          <div key={i} className="overlay-per">
            {(items.length > 1 || i > 0) && <div className="cap viz-caption">img {i + 1}</div>}
            <PointViewer cloud={it} />
            {it.note && (
              <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>{it.note}</div>
            )}
          </div>
        ),
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// viewer: three.js scene straight from the typed arrays (no GLB, no loader)
// ---------------------------------------------------------------------------

function PointViewer({ cloud }: { cloud: Cloud }) {
  const holderRef = useRef<HTMLDivElement | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    const holder = holderRef.current;
    if (!holder) return;
    let disposed = false;
    let scene: THREE.Scene, camera: THREE.PerspectiveCamera, renderer: THREE.WebGLRenderer,
      controls: OrbitControls, geom: THREE.BufferGeometry, mat: THREE.PointsMaterial, raf = 0;
    try {
      const { positions, colors, center, maxDim } = cloud;
      scene = new THREE.Scene();
      scene.background = new THREE.Color("#0a0d11");
      camera = new THREE.PerspectiveCamera(
        50, holder.clientWidth / holder.clientHeight, maxDim / 1e4, maxDim * 1e4);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(holder.clientWidth, holder.clientHeight);
      holder.appendChild(renderer.domElement);
      // grid scaled to the scene (a fixed 10-unit grid would be tiny/large)
      scene.add(new THREE.GridHelper(maxDim * 2, 20, 0x2f3b4a, 0x1c232d));

      // the point cloud IS the geometry — typed arrays in, zero encoding
      geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geom.translate(-center[0], -center[1], -center[2]);   // orbit the middle
      mat = new THREE.PointsMaterial({
        size: Math.max(maxDim / 200, 1e-4),   // ≈2–4 px at fit distance
        vertexColors: true,
        sizeAttenuation: true,
      });
      scene.add(new THREE.Points(geom, mat));

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      camera.position.set(0, maxDim * 0.25, maxDim * 1.6);
      camera.lookAt(0, 0, 0);
      controls.target.set(0, 0, 0);
      controls.update();
    } catch (e) {
      if (!disposed) setFailed(String((e as Error).message || e));
      return;
    }

    const loop = () => {
      if (disposed) return;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    const ro = new ResizeObserver(() => {
      if (disposed || !holder.clientWidth) return;
      camera.aspect = holder.clientWidth / holder.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(holder.clientWidth, holder.clientHeight);
    });
    ro.observe(holder);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      holder.removeChild(renderer.domElement);
      geom.dispose();
      mat.dispose();
    };
  }, [cloud]);

  return (
    <div>
      <div className="glbview" ref={holderRef} />
      {failed && <div className="note" style={{ color: "var(--err)" }}>
        point cloud render failed: {failed}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// per-item build: depth + intrinsics (+ base-image colors) or points fallback
// ---------------------------------------------------------------------------

/** Materialize one item cell (``item[prop]``) as numbers: inline arrays stay
 *  inline; buffer artifacts are fetched (this visualizer must render them). */
async function cellOf(
  item: unknown, prop: string,
): Promise<{ data: Float32Array | Float64Array | Uint8Array; shape: number[] } | null> {
  const cell = item && typeof item === "object" && !Array.isArray(item)
    ? (item as Record<string, unknown>)[prop]
    : undefined;
  if (cell === undefined || cell === null) return null;
  const t = await fetchTyped(cell as SerValue);
  return t && t.data.length > 0 ? t : null;
}

async function buildItem(item: unknown, cfg: Cfg): Promise<Cloud> {
  const notes: string[] = [];

  // ---- 1) depth map (+ intrinsics) -> back-project every valid pixel ----
  if (cfg.depth) {
    const d = await cellOf(item, cfg.depth);
    if (d && d.shape.length === 2) {
      const H = Number(d.shape[0]), W = Number(d.shape[1]);
      if (H > 0 && W > 0 && d.data.length >= H * W) {
        const K = cfg.intrinsics ? await cellOf(item, cfg.intrinsics) : null;
        if (K && K.data.length >= 6) {
          const fx = Number(K.data[0]), cx = Number(K.data[2]);
          const fy = Number(K.data[4]), cy = Number(K.data[5]);
          if (![fx, fy].some((f) => !Number.isFinite(f) || f === 0)) {
            const mask = cfg.mask ? await cellOf(item, cfg.mask) : null;
            const img = cfg.baseUrl ? await loadPixels(cfg.baseUrl) : null;
            const pixel = cfg.projection === "pixel";
            const N = H * W;
            const P = new Float32Array(N * 3);
            const C = new Float32Array(N * 3);
            let M = 0, bad = 0;
            for (let v = 0; v < H; v++) {
              const uy = (v + 0.5) / (pixel ? 1 : H);
              const py = img ? clampIdx((v + 0.5) * img.h / H, img.h) : -1;
              for (let u = 0; u < W; u++) {
                const k = v * W + u;
                const z = Number(d.data[k]);
                if (!Number.isFinite(z) || z <= 0) { bad++; continue; }
                if (mask && mask.data.length >= N && Number(mask.data[k]) <= 0.5) { bad++; continue; }
                const ux = (u + 0.5) / (pixel ? 1 : W);
                P[M * 3] = z * (ux - cx) / fx;
                P[M * 3 + 1] = z * (uy - cy) / fy;
                P[M * 3 + 2] = z;
                if (img) {
                  const px = clampIdx((u + 0.5) * img.w / W, img.w);
                  const o = (py * img.w + px) * 4;
                  C[M * 3] = img.data[o] / 255;
                  C[M * 3 + 1] = img.data[o + 1] / 255;
                  C[M * 3 + 2] = img.data[o + 2] / 255;
                }
                M++;
              }
            }
            if (M === 0) throw new Error("no valid pixels after depth/mask filtering");
            if (!img) {
              colorByXyzRange(P, C, M);     // no base image: keep "axis" readable
              notes.push("no base image → colored by xyz range");
            }
            const cloud = finish(P, C, M);
            notes.unshift(`${M.toLocaleString()} of ${N.toLocaleString()} px reprojected from depth` +
              (bad ? ` · ${bad.toLocaleString()} dropped (invalid depth/mask)` : "") +
              (img ? " · colored from the input image" : ""));
            return { ...cloud, note: notes.join(" · ") };
          }
        }
      }
    }
  }

  // ---- 2) fallback: box-provided (…, 3) positions, xyz-range colors ----
  if (cfg.points) {
    const p = await cellOf(item, cfg.points);
    if (p && p.shape.length >= 2 && Number(p.shape[p.shape.length - 1]) === 3) {
      const grid = p.shape.slice(0, -1).map(Number);
      const N = grid.reduce((a, b) => a * b, 1);
      if (N > 0 && p.data.length >= N * 3) {
        const mask = cfg.mask ? await cellOf(item, cfg.mask) : null;
        const kept: number[] = [];
        let bad = 0;
        for (let i = 0; i < N; i++) {
          if (mask && mask.data.length >= N && Number(mask.data[i]) <= 0.5) { bad++; continue; }
          if (!Number.isFinite(Number(p.data[i * 3])) ||
              !Number.isFinite(Number(p.data[i * 3 + 1])) ||
              !Number.isFinite(Number(p.data[i * 3 + 2]))) { bad++; continue; }
          kept.push(i);
        }
        if (kept.length === 0) throw new Error("no valid points");
        const M = kept.length;
        const P = new Float32Array(M * 3);
        for (let j = 0; j < M; j++) {
          for (let c = 0; c < 3; c++) P[j * 3 + c] = Number(p.data[kept[j] * 3 + c]);
        }
        const C = new Float32Array(M * 3);
        colorByXyzRange(P, C, M);
        const cloud = finish(P, C, M);
        return { ...cloud, note: `${M.toLocaleString()} of ${N.toLocaleString()} points` +
          (mask ? " (masked" : "") + (bad ? ` + ${bad} non-finite` : "") +
          (mask ? ")" : " (non-finite)") + " dropped · colored by xyz range" };
      }
    }
  }

  throw new Error(cfg.depth
    ? `“${cfg.depth}” has no reprojectable grid (needed (H, W) depth + (3, 3) intrinsics${cfg.intrinsics ? "" : " — params.intrinsics missing"})`
    : `“${cfg.points ?? ""}” is not (…, 3) positions`);
}

/** Center + extent from the (already valid) positions. */
function finish(P: Float32Array, C: Float32Array, M: number): Cloud {
  const mn: number[] = [Infinity, Infinity, Infinity];
  const mx: number[] = [-Infinity, -Infinity, -Infinity];
  for (let j = 0; j < M * 3; j++) {
    const c = j % 3, x = P[j];
    if (x < mn[c]) mn[c] = x;
    if (x > mx[c]) mx[c] = x;
  }
  return {
    positions: P, colors: C,
    center: [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2],
    maxDim: Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-6),
    note: "",
  };
}

function clampIdx(x: number, len: number): number {
  return Math.min(len - 1, Math.max(0, Math.floor(x)));
}

/** per-channel min/max normalization into 0..1 (the "which axis" fallback). */
function colorByXyzRange(P: Float32Array, C: Float32Array, M: number): void {
  const mn = [Infinity, Infinity, Infinity];
  const mx = [-Infinity, -Infinity, -Infinity];
  for (let j = 0; j < M * 3; j++) {
    const c = j % 3, x = P[j];
    if (x < mn[c]) mn[c] = x;
    if (x > mx[c]) mx[c] = x;
  }
  for (let j = 0; j < M * 3; j++) {
    const c = j % 3;
    C[j] = (P[j] - mn[c]) / ((mx[c] - mn[c]) || 1);
  }
}

/** Decode a (same-origin) image to raw RGB, downscaled past a 4096² cap so
 *  8K uploads can't melt the tab.  Returns null on any failure (then the
 *  visualizer degrades to xyz-range colors, never breaks). */
async function loadPixels(
  url: string,
): Promise<{ data: Uint8ClampedArray; w: number; h: number } | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    const blob = await r.blob();
    const src = URL.createObjectURL(blob);
    const img = new Image();
    img.src = src;
    await img.decode();
    let w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) { URL.revokeObjectURL(src); return null; }
    const CAP = 4096 * 4096;
    if (w * h > CAP) {
      const s = Math.sqrt(CAP / (w * h));
      w = Math.max(1, Math.floor(w * s));
      h = Math.max(1, Math.floor(h * s));
    }
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const ctx = off.getContext("2d", { willReadFrequently: true });
    if (!ctx) { URL.revokeObjectURL(src); return null; }
    ctx.drawImage(img, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    URL.revokeObjectURL(src);
    return { data, w, h };
  } catch {
    return null;
  }
}
