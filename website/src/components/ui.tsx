import katex from 'katex';
import { ReactNode, useEffect, useId, useMemo, useRef, useState } from 'react';
import { Gray } from '../lib/imageOps';
import { ColormapName, grayToImageData, overlayImageData } from '../lib/colormaps';

/* ---------------- maths ---------------- */
export function Tex({ children, display = false }: { children: string; display?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: display, throwOnError: false, strict: 'ignore' }), [children, display]);
  return display ? <div className="tex-display" dangerouslySetInnerHTML={{ __html: html }} /> : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/* ---------------- provenance ---------------- */
export type ProvKind = 'live' | 'model' | 'inference' | 'illustrative' | 'missing';
const PROV_TEXT: Record<ProvKind, [string, string]> = {
  live: ['Live simulation', 'Computed in your browser from exact mathematics or image processing. No trained model is involved.'],
  model: ['Model output', 'Precomputed by a trained checkpoint through the export pipeline. Hover the figure details for checkpoint and script provenance.'],
  inference: ['Live inference', 'Computed now by the local Python inference server using a trained checkpoint.'],
  illustrative: ['Illustrative', 'A toy or hypothetical example built to show a mechanism. It is not a result about ultrasound models.'],
  missing: ['Awaiting artifact', 'This view needs outputs from a trained model that have not been exported yet.'],
};
export function Prov({ kind, detail }: { kind: ProvKind; detail?: string }) {
  const [label, text] = PROV_TEXT[kind];
  return <span className={`prov ${kind}`} title={detail ? `${text}\n\n${detail}` : text}>{label}</span>;
}
export function ProvLegend() {
  return (
    <div className="prov-legend">
      {(['live', 'model', 'inference', 'illustrative', 'missing'] as ProvKind[]).map((k) => (
        <span key={k} className="row" style={{ gap: '0.45rem' }}><Prov kind={k} /><span className="muted small">{PROV_TEXT[k][1].split('.')[0]}.</span></span>
      ))}
    </div>
  );
}

/* ---------------- figures ---------------- */
export function Figure({ title, prov, provDetail, caption, children, wide = true, id }: {
  title: string; prov: ProvKind | ProvKind[]; provDetail?: string; caption?: ReactNode; children: ReactNode; wide?: boolean; id?: string;
}) {
  const kinds = Array.isArray(prov) ? prov : [prov];
  return (
    <figure className={`figure ${wide ? 'l-page' : ''}`} id={id} data-beat>
      <div className="figure-body">
        <div className="figure-head">
          <h4 className="figure-title">{title}</h4>
          <div className="row" style={{ gap: '0.4rem' }}>{kinds.map((k) => <Prov key={k} kind={k} detail={provDetail} />)}</div>
        </div>
        {children}
      </div>
      {caption && <figcaption className="figure-caption">{caption}</figcaption>}
    </figure>
  );
}

export function Empty({ title, children }: { title: string; children: ReactNode }) {
  return <div className="empty" role="status"><strong>{title}</strong>{children}</div>;
}

export function Deep({ title, children, kind = 'math' }: { title?: string; children: ReactNode; kind?: 'math' | 'code' | 'note' }) {
  const t = title ?? (kind === 'math' ? 'Mathematical details' : kind === 'code' ? 'View PyTorch implementation' : 'More detail');
  return (
    <details className="deep">
      <summary>{t}</summary>
      <div className="deep-body">{children}</div>
    </details>
  );
}

/** Lightweight Python highlighter: comments and a few keywords. */
export function Code({ children }: { children: string }) {
  const html = useMemo(() => {
    const esc = children.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return esc.split('\n').map((line) => {
      const ci = line.indexOf('#');
      const head = ci >= 0 ? line.slice(0, ci) : line;
      const tail = ci >= 0 ? `<span class="c">${line.slice(ci)}</span>` : '';
      return head.replace(/\b(def|class|return|import|from|for|in|if|else|with|as|None|True|False|lambda|yield)\b/g, '<span class="k">$1</span>') + tail;
    }).join('\n');
  }, [children]);
  return <pre className="code"><code dangerouslySetInnerHTML={{ __html: html }} /></pre>;
}

/* ---------------- controls ---------------- */
export function Slider({ label, value, min, max, step = 1, onChange, format, disabled, ticks }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void;
  format?: (v: number) => string; disabled?: boolean; ticks?: number[];
}) {
  const id = useId();
  return (
    <div className="control">
      <label className="control-label" htmlFor={id}><span>{label}</span><output htmlFor={id}>{format ? format(value) : value}</output></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        list={ticks ? id + '-ticks' : undefined} onChange={(e) => onChange(Number(e.target.value))} />
      {ticks && <datalist id={id + '-ticks'}>{ticks.map((t) => <option key={t} value={t} />)}</datalist>}
    </div>
  );
}

export function Seg<T extends string | number>({ options, value, onChange, label, disabled }: {
  options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[]; value: T; onChange: (v: T) => void; label: string; disabled?: boolean;
}) {
  return (
    <div className="control">
      <span className="control-label"><span>{label}</span></span>
      <div className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={String(o.value)} type="button" aria-pressed={o.value === value} disabled={disabled || o.disabled} title={o.title} onClick={() => onChange(o.value)}>{o.label}</button>
        ))}
      </div>
    </div>
  );
}

/* ---------------- canvases ---------------- */
export function GrayCanvas({ img, cmap = 'gray', label, pixelated, onClick, overlay, style, ariaLabel, children }: {
  img: Gray; cmap?: ColormapName; label?: string; pixelated?: boolean;
  onClick?: (x: number, y: number) => void;
  overlay?: { heat: Gray; opacity: number; cmap: ColormapName; threshold?: number };
  style?: React.CSSProperties; ariaLabel?: string; children?: ReactNode;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = img.w; c.height = img.h;
    const ctx = c.getContext('2d')!;
    ctx.putImageData(overlay ? overlayImageData(img, overlay.heat, overlay.opacity, overlay.cmap, overlay.threshold) : grayToImageData(img, cmap), 0, 0);
  }, [img, cmap, overlay]);
  return (
    <div className={`well ${pixelated ? 'pixelated' : ''}`} style={style}>
      <canvas ref={ref} role="img" aria-label={ariaLabel ?? label ?? 'image'}
        style={{ cursor: onClick ? 'crosshair' : undefined }}
        onClick={onClick ? (e) => {
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          onClick(((e.clientX - r.left) / r.width) * img.w, ((e.clientY - r.top) / r.height) * img.h);
        } : undefined} />
      {children}
      {label && <span className="well-label">{label}</span>}
    </div>
  );
}

/* ---------------- question with reveal ---------------- */
export function Ask({ q, options, reveal, multi = false }: { q: ReactNode; options: string[]; reveal: ReactNode; multi?: boolean }) {
  const [sel, setSel] = useState<string[]>([]);
  const [shown, setShown] = useState(false);
  return (
    <div className="ask" data-beat>
      <p className="ask-q">{q}</p>
      <div className="ask-options">
        {options.map((o) => (
          <button key={o} type="button" aria-pressed={sel.includes(o)} onClick={() => setSel(multi ? (sel.includes(o) ? sel.filter((x) => x !== o) : [...sel, o]) : [o])}>{o}</button>
        ))}
        <button type="button" className="btn" style={{ marginLeft: 'auto' }} onClick={() => setShown(!shown)} aria-expanded={shown}>{shown ? 'Hide discussion' : 'Show discussion'}</button>
      </div>
      {shown && <div className="ask-reveal">{reveal}</div>}
    </div>
  );
}

/* ---------------- misc ---------------- */
export const MODEL_META = {
  densenet161: { name: 'DenseNet-161', color: 'var(--m-densenet)', text: 'var(--m-densenet)' },
  vit_b16: { name: 'ViT-B/16', color: 'var(--m-vit)', text: 'var(--m-vit-text)' },
  usf_mae: { name: 'USF-MAE', color: 'var(--m-usfmae)', text: 'var(--m-usfmae-text)' },
} as const;

export function ModelTag({ id }: { id: keyof typeof MODEL_META }) {
  const m = MODEL_META[id];
  return <span style={{ whiteSpace: 'nowrap', color: m.text, fontWeight: 600 }}><span className="swatch" style={{ background: m.color }} />{m.name}</span>;
}

export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
export const fmtInt = (v: number) => v.toLocaleString('en-US');
export const fmtM = (v: number) => `${(v / 1e6).toFixed(2)} M`;

export function useTooltip() {
  const [tip, setTip] = useState<{ x: number; y: number; html: ReactNode } | null>(null);
  const node = tip ? <div className="tooltip" style={{ left: tip.x + 12, top: tip.y + 12 }}>{tip.html}</div> : null;
  return { show: (e: React.MouseEvent, html: ReactNode) => setTip({ x: e.clientX, y: e.clientY, html }), hide: () => setTip(null), node };
}
