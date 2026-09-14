/** Small shared UI bits. */
import type { ReactNode } from "react";

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinnerbox">
      <span className="spinner" /> {label || "…"}
    </span>
  );
}

export function StatusChip({ status }: { status?: string | null | unknown }) {
  if (!status) return <span className="chip dim">no status</span>;
  const cls =
    status === "done" ? "ok" : status === "error" ? "err" : status === "empty_request" ? "dim" : "warn";
  return <span className={`chip ${cls}`}>{String(status)}</span>;
}

export function ErrorBox({ title, detail }: { title?: string; detail: Record<string, unknown> }) {
  return (
    <div className="errorbox">
      {title && <div className="errorbox-title">{title}</div>}
      <pre>{JSON.stringify(detail, null, 2)}</pre>
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return <div className="note">{children}</div>;
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="sect">
      <div className="sect-title">{children}</div>
      {right && <div className="sect-right">{right}</div>}
    </div>
  );
}

export function bytesShort(n: number): string {
  if (!Number.isFinite(n)) return "?";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

export function fmtNum(n: number): string {
  if (Number.isInteger(n) && Math.abs(n) < 1e6) return String(n);
  return String(Number(n.toPrecision(6)));
}
