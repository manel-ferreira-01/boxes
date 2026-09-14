/** Fleet page: entries table (probe dot, name/addr/def, note), add-entry
 *  form, probe (on-demand + auto on mount, 3 s timeout), open console, delete. */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import type { BoxDef, FleetEntry } from "../api";
import { ErrorBox } from "../ui";

function detailOf(e: unknown): Record<string, unknown> {
  if (e instanceof ApiError) return { status: e.status, ...e.detail };
  return { message: String(e) };
}

function timeAgo(ts: number): string {
  const s = Math.max(0, Math.round((Date.now() - ts * 1000) / 1000));
  if (s < 10) return "just now";
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

function Dot({ entry, probing }: { entry: FleetEntry; probing?: boolean }) {
  const p = entry.last_probe;
  if (probing || !p) {
    const title = probing ? "probing…" : "not probed yet";
    return <span className="dot unknown" title={title} />;
  }
  const cls = p.reachable ? "ok" : "err";
  const title = p.reachable
    ? `reachable${p.reflection ? " · reflection" : ""} — last probe ${timeAgo(p.at)}`
    : `unreachable — ${p.error || "probe failed"} (${timeAgo(p.at)})`;
  return <span className={`dot ${cls}`} title={title} />;
}

export default function FleetPage({ defs }: { defs: BoxDef[] }) {
  const [entries, setEntries] = useState<FleetEntry[] | null>(null);
  const [loadErr, setLoadErr] = useState<Record<string, unknown> | null>(null);
  const [probing, setProbing] = useState<Record<string, boolean>>({});
  const [defId, setDefId] = useState(defs[0]?.id ?? "");
  const [name, setName] = useState("");
  const [addr, setAddr] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [addErr, setAddErr] = useState<Record<string, unknown> | null>(null);
  const autoProbed = useRef(false);

  const defFor = useCallback((id: string | null) => defs.find((d) => d.id === id) ?? null, [defs]);

  const probe = useCallback(async (id: string) => {
    setProbing((p) => ({ ...p, [id]: true }));
    try {
      const updated = await api.fleetProbe(id, 3);
      setEntries((es) => (es ? es.map((e) => (e.id === id ? updated : e)) : es));
    } catch {
      /* unreachable stays red/unknown — the dot reflects state, not errors */
    } finally {
      setProbing((p) => ({ ...p, [id]: false }));
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      setEntries((await api.fleet()).entries);
      setLoadErr(null);
    } catch (e) {
      setLoadErr(detailOf(e));
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  // Auto-probe the never-probed entries once, on mount (3 s timeout each).
  useEffect(() => {
    if (!entries || autoProbed.current) return;
    autoProbed.current = true;
    for (const e of entries) if (!e.last_probe) void probe(e.id);
  }, [entries, probe]);

  function suggestedName(): string {
    const d = defFor(defId);
    return name.trim() || d?.name || defId;
  }

  async function add() {
    if (!addr.trim() || !defId) { setAddErr({ message: "addr (host:port) and def are required" }); return; }
    setBusy(true); setAddErr(null);
    try {
      await api.fleetAdd({ name: suggestedName(), addr: addr.trim(), def_id: defId, note: note.trim() || null });
      setAddr(""); setNote(""); setName("");
      await refresh();
    } catch (e) {
      setAddErr(detailOf(e));
    } finally {
      setBusy(false);
    }
  }

  async function del(e: FleetEntry) {
    if (!window.confirm(`remove ${e.name} (${e.addr})?`)) return;
    await api.fleetDel(e.id).catch(() => undefined);
    await refresh();
  }

  return (
    <div>
      <h2>Fleet</h2>
      <p className="sub">
        Boxes are independent Dockerized gRPC services; the webui stores their
        addresses here and calls them directly (shared <span className="mono">Process</span> envelope).
      </p>

      <div className="panel">
        <h3>Boxes</h3>
        {loadErr && <ErrorBox title="could not load the fleet" detail={loadErr} />}
        {!entries && !loadErr && <div className="empty">loading…</div>}
        {entries && entries.length === 0 && !loadErr && (
          <div className="empty">
            <div className="big">🛰️</div>
            no fleet entries yet — add your boxes below (typical local fleet:
            clip :9061, sbert :9062, tapnext :9063, lang_sam :9064)
          </div>
        )}
        {entries && entries.length > 0 && (
          <table className="list">
            <thead>
              <tr><th></th><th>name</th><th>addr</th><th>def</th><th>note</th><th></th></tr>
            </thead>
            <tbody>
              {entries.map((e) => {
                const d = defFor(e.def_id);
                return (
                  <tr key={e.id}>
                    <td><Dot entry={e} probing={probing[e.id]} /></td>
                    <td>
                      <b>{e.name}</b>
                      {e.last_probe && (
                        <span className="note"> · probed {timeAgo(e.last_probe.at)}</span>
                      )}
                    </td>
                    <td className="mono">{e.addr}</td>
                    <td>
                      {e.def_id
                        ? <a href={`#/box/${e.def_id}`}>{e.def_id}</a>
                        : <span className="note">—</span>}
                      {d?.note && <div className="note">{d.note}</div>}
                    </td>
                    <td>{e.note || ""}</td>
                    <td>
                      <div className="btnrow" style={{ justifyContent: "flex-end" }}>
                        <button className="btn small" disabled={probing[e.id]} onClick={() => void probe(e.id)}>
                          {probing[e.id] ? "probing…" : "probe"}
                        </button>
                        {e.def_id && (
                          <a className="btn small" href={`#/box/${e.def_id}`}>open console →</a>
                        )}
                        <button className="btn small danger" onClick={() => void del(e)}>delete</button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <h3>Add a box</h3>
        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr 1fr .8fr auto", gap: 10, alignItems: "end" }}>
          <label className="fld" style={{ marginBottom: 0 }}>
            <span className="lbl">def <span className="req">*</span></span>
            <select
              value={defId}
              onChange={(e) => { setDefId(e.target.value); setName(""); }}
            >
              {defs.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </label>
          <label className="fld" style={{ marginBottom: 0 }}>
            <span className="lbl">addr <span className="req">*</span></span>
            <input type="text" placeholder="host:port (e.g. 127.0.0.1:9061)"
              value={addr} onChange={(e) => setAddr(e.target.value)} />
          </label>
          <label className="fld" style={{ marginBottom: 0 }}>
            <span className="lbl">name <span className="hint">default: def name</span></span>
            <input type="text" placeholder={suggestedName()} value={name}
              onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="fld" style={{ marginBottom: 0 }}>
            <span className="lbl">note</span>
            <input type="text" value={note} onChange={(e) => setNote(e.target.value)} />
          </label>
          <button className="btn primary" disabled={busy} onClick={() => void add()}>
            {busy ? "adding…" : "add"}
          </button>
        </div>
        {addErr && <div style={{ marginTop: 10 }}><ErrorBox detail={addErr} /></div>}
      </div>
    </div>
  );
}
