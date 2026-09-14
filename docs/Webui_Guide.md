# boxes-webui — guide & working state

> The declarative web layer over the fleet: a box-agnostic core, YAML box
> definitions, a FastAPI HTTP API, and a def-driven SPA.
> **This replaces `webui/STATUS.md`** (retired; see `webui/README.md` for the
> feature documentation — this file is the *working-state/session handoff*).

## 1. What it is

```
   browser / curl ──HTTP──▶ webui (FastAPI, box-agnostic core)
                                  │  boxes_client.Box.run(...)
   box by IP:port (Process Envelope) ── clip · tapnext · lang_sam · sbert · vggt
```

**Design rule (inherited from `boxes_client`):** the core is *smart about
shape, dumb about content*. Nothing in `webui/src/webui/` or `webui/web/src/`
names a box. **All box knowledge lives in `webui/boxes/*.yaml`** — adding a
box = one YAML file, never code. The per-box README under `images/` stays the
authoritative request-shape source; defs link to them via `docs:`.

**Scope is contract-only:** every box is served through the one shared
`Process` RPC. `yologpt` and `opencv_box` are deliberately excluded (pre-
contract RPCs); re-adding them later = new YAML defs only (`build_call`
refuses non-`Process` methods with a clear error until then).

## 2. Current state (verified)

| Layer | State |
|---|---|
| Backend | **48/48 tests green** (`python3 -m pytest tests/ -q` from `webui/`), live-verified against the running fleet (clip, lang_sam, tapnext; error paths 400/502) |
| Frontend | `tsc --noEmit && vite build` clean; `web/dist` auto-mounted by the FastAPI app (API + `/docs` keep priority) |
| Session fixes applied | ✅ tab-switch state leakage (console now remounts per def), ✅ video input for tapnext (`video_frames` widget), ✅ tapnext tracks `(y,x)` order corrected + per-frame visibility toggle, ✅ labeled/legend heatmaps (clip), ✅ input mosaic |

Still **unverified in-browser / live**: vggt GLB (no vggt box in the local
fleet — camera auto-fit pending), pixel-level pass of `overlay` and the 3-D
`glb` orbit. yologpt/opencv intentionally out of scope.

## 3. Run / build / test loop

```bash
# deps (client first, editable)
pip install -e boxes_client && pip install -e webui

# backend tests
cd webui && python3 -m pytest tests/ -q            # 48 passed in ~2.5 s

# frontend: typecheck + build (dist/ is served by the app)
cd webui/web && npx tsc --noEmit && npx vite build  # warn: three.js >500 kB chunk (cosmetic)

# run (SPA served from web/dist; root "/" = API health JSON, open /index.html#/…)
WEBUI_DATA_DIR=$PWD/data WEBUI_PORT=8090 \
  python3 -m uvicorn webui.app:factory --host 127.0.0.1 --port 8090
```

Fleet seed (local docker fleet): `clip 9061 · sbert 9062 · tapnext 9063 ·
lang_sam 9064` (see `webui/data/fleet.json`).

## 4. Layout

```
webui/
├── boxes/                    # ← the ONLY per-box knowledge in the whole webui
│   ├── clip.yaml  lang_sam.yaml  sbert.yaml  tapnext.yaml  vggt.yaml
├── src/webui/
│   ├── app.py                # FastAPI factory (env-driven; mounts web/dist)
│   ├── config.py             # WEBUI_* env, defaults
│   ├── core/                 # box-agnostic
│   │   ├── schema.py         # Pydantic models + widget/visualizer vocab
│   │   ├── registry.py       # load+validate boxes/*.yaml (fail fast)
│   │   ├── caller.py         # build_call() pure / execute() wire (only gRPC)
│   │   ├── serialize.py      # Result -> JSON + artifacts (never raises)
│   │   ├── artifact.py       # token store (TTL + cap)
│   │   └── fleet.py          # fleet.json CRUD + Box.info() probes
│   └── api/                  # routes: defs / fleet / upload / file / call
├── web/                      # SPA (React + TS + Vite)
│   └── src/
│       ├── App.tsx           # hash router + topbar (fleet + tab per def)
│       ├── pages/            # FleetPage (probe dots, add/delete) + ConsolePage
│       ├── form/widgets.tsx  # def-driven form widgets
│       ├── viz/              # json/table/image_grid/overlay/matrix/tensor/
│       │                     #   glb/tracks_player/download
│       ├── resolvers.ts      # inline/buffer/file artifact → typed values
│       └── api.ts            # REST client + wire types
└── tests/                    # registry · caller · API e2e (fake boxes, real gRPC)
```

## 5. The contract the UI renders from

* **widgets** — `image_upload · video_frames · file_upload · tags ·
  text_repeat · slider · select · number · json`
* **visualizers** — `json (fallback) · table · image_grid · overlay
  (box/mask/point/flow) · matrix · tensor · glb · tracks_player · download`
* result field `"*"` = wildcard fallback, so the UI can never get stuck on a
  field a definition forgot.

Def-driven extras (generic, no box names in code):

* **`video_frames` widget** — video file (mp4/webm/mov) or images. Video →
  N evenly-spaced frames (8–128, user-selected) extracted in the *browser*
  (seek+canvas, ≤960 px, JPEG), each uploaded as an image `"@token"`; wire
  shape is the same image list, so boxes never know a video happened. Used
  by tapnext (the box tracks the whole list in one call/session).
* **`matrix` result `params`** — `{row_labels: <input field>, col_labels:
  <input field>}` names the form fields that label the axes. Labels are
  snapshotted **per call** into call history (so history re-renders stay
  correct); image fields label rows as `img 1..n`. Heatmap: viridis color
  scale + min→max legend, axis labels (rotated when wide, all kept),
  hover readout `row × col → value`. Used by clip.
* **`tracks_player`** — steps assembled from the console's call history:
  one step per uploaded frame, points from `tracks (F, T, 2)` **as (y, x) —
  the box contract** (webui flips to (x,y) for drawing), visibility from
  `visibles` **per frame** — points flagged invisible are dropped (and
  trails into them stop). Session quick-actions (`reset`/`list`) available
  from the def's `session.actions`.

## 6. Wire rules & serialization (what the SPA must handle)

* files travel as `"@<token>"` refs (uploaded first) → `bytes` in the
  envelope; bare strings in `data` are always literals;
* request validated **against the def** — unknown params/section/fields/
  commands → 400 with `known: [...]`;
* reset semantics: `command: reset` *is* the reset; stateful boxes are never
  auto-reset;
* serialized values (`core/serialize.py`, never raises — degrades):
  - `{kind:"array", dtype, shape, values}` — inline (≤ 65 536 elements)
  - `{kind:"buffer", url, dtype, shape, size}` — fetch → typed buffer
  - `{kind:"file", url, mime, size}` — images/GLB/etc.
    (sniffed: `glTF` → `model/gltf-binary`, JPEG/PNG/MP4)
  - everything else plain JSON (exotics → pickle + note).

## 7. Environment gotchas (learned the hard way)

* **Backend changes need a restart** — defs load at startup. Kill by port
  (`ss -ltnp | grep 8090 | grep -oP 'pid=\K[0-9]+' | xargs -r kill`);
  do **not** `pkill -f "uvicorn webui"` (matches your own shell).
* Every frontend change needs `npx tsc --noEmit && npx vite build` (the
  served app is `web/dist`) — and a **hard reload** in the browser to beat
  the old JS bundle.
* `tapnext` wants `data.images` as a **list** (even single frame) — the box
  checks `isinstance`.
* First GPU call on a box takes a few seconds (CPU→GPU model migration);
  the API `timeout` (s) is honored per call.
* tapnext sessions live server-side and are reaped after
  `TAPNEXT_SESSION_TTL` (default 1800 s) — use the console **Reset** /
  **regenerate session** before restarting a sequence.
* lang_sam decoded item keys: `boxes / masks / scores / text_labels /
  mask_scores` (a live test caught an early `bbox` typo).
* test asset: `images/clip/test/dog.jpg`.

## 8. Next steps (open)

1. vggt GLB: camera auto-fit confirmation on a live reconstruction (no vggt
   box in the local fleet yet).
2. In-browser human pass: `overlay` pixel check, glb orbit, history
   click-through (all code-built and data-verified, just not eyeballed).
3. Optional: side-by-side prompts on lang_sam; `image_grid` for yologpt
   once it migrates to the envelope.
4. Optional: code-split the three.js chunk (currently one ~780 kB bundle).
