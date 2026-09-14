/** Image grid: value is a file ref, a list of file refs, or a list of
 *  inline image bytes refs. */
import { isRef } from "../resolvers";

interface Item {
  url: string | null;
  note?: string;
}

export function ImageGrid({ value, title }: { value: unknown; title?: string }) {
  const items: Item[] = collect(value);
  if (!items.length) return <div className="note">no images in this field</div>;
  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      <div className="imggrid">
        {items.map((it, i) => (
          <figure key={i}>
            {it.url
              ? <img src={it.url} alt={`image ${i}`} loading="lazy" />
              : <div className="note">no url</div>}
            {(it.note || i > 0) && <figcaption>{i}: {it.note || ""}</figcaption>}
          </figure>
        ))}
      </div>
    </div>
  );
}

function collect(value: unknown): Item[] {
  if (value === null || value === undefined) return [];
  if (isRef(value)) return [ref(value)];
  if (Array.isArray(value)) return value.map((x) => itemFrom(x));
  return [itemFrom(value)];
}

function ref(v: unknown): Item {
  const r = v as { url?: string; mime?: string; size?: number };
  return { url: typeof r.url === "string" ? r.url : null, note: `${r.mime || ""} ${typeof r.size === "number" ? Math.round(r.size / 1024) + "kB" : ""}`.trim() };
}

function itemFrom(x: unknown): Item {
  if (isRef(x)) return ref(x);
  if (typeof x === "string" && x.length > 0 && !x.startsWith("@")) return { url: x };
  return { url: null };
}
