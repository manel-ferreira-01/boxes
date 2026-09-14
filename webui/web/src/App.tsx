/** App shell: hash router (#/fleet, #/box/<defId>) + topbar nav.
 *  Nothing here knows a box's request shape — the defs drive the pages. */
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, ApiError } from "./api";
import type { BoxDef } from "./api";
import { ErrorBox, Spinner } from "./ui";
import FleetPage from "./pages/FleetPage";
import ConsolePage from "./pages/ConsolePage";

function useHashRoute(): string {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const f = () => setHash(window.location.hash);
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return hash;
}

export function routeOf(hash: string): { page: "fleet" } | { page: "box"; id: string } {
  const path = hash.replace(/^#/, "") || "/";
  const m = path.match(/^\/box\/([^/]+)/);
  if (m) return { page: "box", id: decodeURIComponent(m[1]) };
  return { page: "fleet" };
}

export default function App() {
  const hash = useHashRoute();
  const route = routeOf(hash);
  const [defs, setDefs] = useState<BoxDef[] | null>(null);
  const [err, setErr] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    api.defs()
      .then((d) => setDefs(d.defs))
      .catch((e) => {
        setErr(e instanceof ApiError ? { status: e.status, ...e.detail } : { message: String(e) });
      });
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [hash]);

  let content: ReactNode;
  if (err) content = <ErrorBox title="could not load box definitions" detail={err} />;
  else if (!defs) content = <div className="empty"><Spinner label="loading definitions…" /></div>;
  else if (route.page === "box") content = <ConsolePage defs={defs} defId={route.id} />;
  else content = <FleetPage defs={defs} />;

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="#/fleet" style={{ color: "var(--fg)", textDecoration: "none" }}>
          📦 boxes <small>webui · contract-only fleet console</small>
        </a>
        {defs && (
          <nav>
            <a href="#/fleet" className={route.page === "fleet" ? "active" : ""}>fleet</a>
            {defs.map((d) => (
              <a
                key={d.id}
                href={`#/box/${d.id}`}
                className={route.page === "box" && route.id === d.id ? "active" : ""}
                title={d.name}
              >
                {d.id}
              </a>
            ))}
          </nav>
        )}
      </header>
      <main>
        <div className="page">{content}</div>
      </main>
    </div>
  );
}
