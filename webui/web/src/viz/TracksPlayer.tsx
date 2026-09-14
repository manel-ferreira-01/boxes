/** Tracks player: steps (frame + tracked points) with prev/next/play and
 *  per-point-id coloring + trails between steps. */
import { useCallback, useEffect, useRef, useState } from "react";
import { PALETTE } from "../resolvers";

export interface TrackStep {
  frameUrl: string | null;
  points?: { x: number; y: number; visible?: boolean }[];
  label?: string;
}

export function TracksPlayer({ steps, title }: { steps: TrackStep[]; title?: string }) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600); // ms per step
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);

  const n = Math.max(1, steps.length);
  const step = steps[Math.min(idx, n - 1)];

  const go = useCallback((i: number) => setIdx(((i % n) + n) % n), [n]);

  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => setIdx((i) => (i + 1) % n), speed);
    return () => window.clearInterval(t);
  }, [playing, n, speed]);

  useEffect(() => {
    let alive = true;
    (async () => {
      const cv = canvasRef.current;
      if (!cv) return;
      const ctx = cv.getContext("2d");
      if (!ctx) return;

      // load the base image once, size the canvas FIRST (resizing clears the
      // canvas), then draw everything in a single pass
      let W = 640, H = 480;
      let img: HTMLImageElement | null = null;
      if (step?.frameUrl) {
        img = await new Promise<HTMLImageElement | null>((res) => {
          const im = new Image();
          im.onload = () => res(im);
          im.onerror = () => res(null);
          im.src = step.frameUrl as string;
        });
        if (img) { W = img.naturalWidth; H = img.naturalHeight; }
      }
      cv.width = W; cv.height = H;
      if (img) ctx.drawImage(img, 0, 0, W, H);
      else {
        ctx.fillStyle = "#10151c";
        ctx.fillRect(0, 0, W, H);
      }
      if (!alive) return;

      // trails from previous step
      const prev = steps[Math.min(idx - 1, n - 1)];
      const pts = step?.points || [];
      if (prev && idx > 0) {
        prev.points?.forEach((p, id) => {
          const cur = pts[id];
          if (!cur || p.visible === false || cur.visible === false) return;
          const color = PALETTE[id % PALETTE.length];
          ctx.strokeStyle = color + "88";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(cur.x, cur.y);
          ctx.stroke();
        });
      }
      pts.forEach((p, id) => {
        if (p.visible === false) return;
        const color = PALETTE[id % PALETTE.length];
        ctx.fillStyle = color;
        ctx.strokeStyle = "#0b0e12";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
      if (alive) setReady(true);
    })();
    return () => { alive = false; };
  }, [step, idx, steps, n]);

  if (!steps.length) {
    return (
      <div>
        {title && <div className="viz-caption">{title}</div>}
        <div className="note">no tracked steps yet — call the box per frame (same session_id) and steps accumulate here</div>
      </div>
    );
  }

  return (
    <div className="player">
      {title && <div className="viz-caption">{title}</div>}
      <div className="stage">
        <canvas ref={canvasRef} style={{ visibility: ready ? "visible" : "hidden" }} />
      </div>
      <div className="controls">
        <button className="btn small" onClick={() => go(idx - 1)}>⏮ prev</button>
        <button className="btn small" onClick={() => go(idx + 1)}>next ⏭</button>
        <button className={`btn small ${playing ? "primary" : ""}`} onClick={() => setPlaying((p) => !p)}>
          {playing ? "❚❚ pause" : "▶ play"}
        </button>
        <label className="step">
          speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} style={{ width: 90, padding: "2px 6px" }}>
            <option value={250}>fast</option>
            <option value={600}>normal</option>
            <option value={1500}>slow</option>
          </select>
        </label>
        <span className="step">step {Math.min(idx, n - 1) + 1} / {n}{step?.label ? ` · ${step.label}` : ""}</span>
        <span className="step" style={{ marginLeft: "auto" }}>{(step?.points?.filter((p) => p.visible !== false).length ?? 0)} points visible</span>
      </div>
      <div className="legend">
        {(step?.points?.length ?? 0) > 0 && (step?.points ?? []).slice(0, 8).map((_, id) => (
          <span key={id}>
            <span className="sw" style={{ background: PALETTE[id % PALETTE.length] }} />
            point {id}
          </span>
        ))}
      </div>
    </div>
  );
}
