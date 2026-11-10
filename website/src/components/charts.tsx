import * as d3 from 'd3';
import { ReactNode } from 'react';
import { useTooltip } from './ui';

/** Small React-rendered SVG charts built on d3 scales/shapes. */

export function Histogram({ bins, width = 300, height = 90, color = 'var(--ink-2)', label, xDomain = [0, 1], compare }: {
  bins: number[]; width?: number; height?: number; color?: string; label?: string; xDomain?: [number, number]; compare?: number[];
}) {
  const m = { t: 6, r: 6, b: 20, l: 6 };
  const mx = Math.max(1, ...bins, ...(compare ?? []));
  const x = d3.scaleLinear().domain([0, bins.length]).range([m.l, width - m.r]);
  const y = d3.scaleSqrt().domain([0, mx]).range([height - m.b, m.t]);
  const area = (b: number[]) => d3.area<number>().x((_, i) => x(i + 0.5)).y0(height - m.b).y1((v) => y(v)).curve(d3.curveStepAfter)(b) ?? '';
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={label ?? 'intensity histogram'}>
      {compare && <path d={area(compare)} fill="none" stroke="var(--rule-strong)" strokeWidth={1} strokeDasharray="3 2" />}
      <path d={area(bins)} fill={color} fillOpacity={0.22} stroke={color} strokeWidth={1.2} />
      <line x1={m.l} x2={width - m.r} y1={height - m.b} y2={height - m.b} stroke="var(--rule-strong)" />
      <text x={m.l} y={height - 5}>{xDomain[0].toFixed(xDomain[0] % 1 ? 1 : 0)}</text>
      <text x={width - m.r} y={height - 5} textAnchor="end">{xDomain[1].toFixed(xDomain[1] % 1 ? 1 : 0)}</text>
      {label && <text x={width / 2} y={height - 5} textAnchor="middle" className="axis-title">{label}</text>}
    </svg>
  );
}

export function HBars({ items, width = 520, max, format = (v: number) => String(v), barColor = 'var(--ink-2)', note }: {
  items: { label: string; value: number; color?: string; sub?: string }[]; width?: number; max?: number; format?: (v: number) => string; barColor?: string; note?: ReactNode;
}) {
  const rowH = 26, lw = 128, rw = 72;
  const h = items.length * rowH + 8;
  const mx = max ?? Math.max(...items.map((i) => i.value), 1e-9);
  const x = d3.scaleLinear().domain([0, mx]).range([0, width - lw - rw]);
  return (
    <svg viewBox={`0 0 ${width} ${h}`} width="100%" role="img" aria-label="bar chart">
      {items.map((it, i) => (
        <g key={it.label} transform={`translate(0, ${i * rowH + 4})`}>
          <text x={lw - 8} y={rowH / 2 + 4} textAnchor="end" style={{ fill: 'var(--ink)' }}>{it.label}</text>
          <rect x={lw} y={4} width={Math.max(0, x(it.value))} height={rowH - 9} fill={it.color ?? barColor} rx={2} />
          <text x={lw + Math.max(0, x(it.value)) + 6} y={rowH / 2 + 4} style={{ fill: 'var(--ink-2)', fontVariantNumeric: 'tabular-nums' }}>{format(it.value)}{it.sub ? `  ${it.sub}` : ''}</text>
        </g>
      ))}
      {note}
    </svg>
  );
}

export interface Series { name: string; color: string; x: number[]; y: number[]; dashed?: boolean; band?: [number[], number[]]; }

export function LineChart({ series, width = 560, height = 240, xLabel, yLabel, yDomain, xDomain, xLog, xTickFormat, yTickFormat, diagonal }: {
  series: Series[]; width?: number; height?: number; xLabel?: string; yLabel?: string; yDomain?: [number, number]; xDomain?: [number, number];
  xLog?: boolean; xTickFormat?: (v: number) => string; yTickFormat?: (v: number) => string; diagonal?: boolean;
}) {
  const tip = useTooltip();
  const m = { t: 12, r: 14, b: 38, l: 48 };
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => [...s.y, ...(s.band ? [...s.band[0], ...s.band[1]] : [])]);
  let xd = (xDomain ?? d3.extent(xs)) as [number, number];
  if (!Number.isFinite(xd[0]) || !Number.isFinite(xd[1])) xd = [0, 1];
  if (xd[0] === xd[1]) xd = xLog ? [xd[0] / 2, xd[1] * 2] : [xd[0] - 1, xd[1] + 1];
  const x = (xLog ? d3.scaleLog() : d3.scaleLinear()).domain(xd).range([m.l, width - m.r]);
  let yd = (yDomain ?? d3.extent(ys.filter(Number.isFinite))) as [number, number];
  if (!Number.isFinite(yd[0]) || !Number.isFinite(yd[1])) yd = [0, 1];
  if (yd[0] === yd[1]) yd = [yd[0] - 0.5, yd[1] + 0.5];
  const y = d3.scaleLinear().domain(yd).nice().range([height - m.b, m.t]);
  const line = (s: Series) => d3.line<number>().x((_, i) => x(s.x[i])).y((v) => y(v)).defined((v) => Number.isFinite(v))(s.y) ?? '';
  const band = (s: Series) => s.band ? d3.area<number>().x((_, i) => x(s.x[i])).y0((_, i) => y(s.band![0][i])).y1((_, i) => y(s.band![1][i]))(s.x) ?? '' : '';
  const xt = xLog ? x.ticks(5) : x.ticks(6);
  return (
    <div style={{ position: 'relative', maxWidth: width * 1.25 }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={`${yLabel ?? ''} versus ${xLabel ?? ''}`}>
        <g className="grid">{y.ticks(5).map((t) => <line key={t} x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} />)}</g>
        {diagonal && <line x1={x(xd[0])} y1={y(xd[0])} x2={x(xd[1])} y2={y(xd[1])} stroke="var(--rule-strong)" strokeDasharray="4 3" />}
        {series.map((s) => s.band && s.band[0].every(Number.isFinite) && <path key={s.name + 'b'} d={band(s)} fill={s.color} fillOpacity={0.13} />)}
        {series.map((s) => <path key={s.name} d={line(s)} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? '5 4' : undefined} />)}
        {series.map((s) => s.x.map((xv, i) => Number.isFinite(s.y[i]) && (
          <circle key={s.name + i} cx={x(xv)} cy={y(s.y[i])} r={3.2} fill={s.color}
            onMouseMove={(e) => tip.show(e, <>{s.name}<br />{xLabel}: {xTickFormat ? xTickFormat(xv) : xv}<br />{yLabel}: {yTickFormat ? yTickFormat(s.y[i]) : s.y[i].toFixed(3)}</>)}
            onMouseLeave={tip.hide} />
        )))}
        <g className="axis">
          <line x1={m.l} x2={width - m.r} y1={height - m.b} y2={height - m.b} />
          {xt.map((t) => <text key={t} x={x(t)} y={height - m.b + 15} textAnchor="middle">{xTickFormat ? xTickFormat(t) : t}</text>)}
          {y.ticks(5).map((t) => <text key={t} x={m.l - 6} y={y(t) + 4} textAnchor="end">{yTickFormat ? yTickFormat(t) : t}</text>)}
        </g>
        {xLabel && <text className="axis-title" x={(m.l + width - m.r) / 2} y={height - 4} textAnchor="middle">{xLabel}</text>}
        {yLabel && <text className="axis-title" transform={`translate(12, ${(m.t + height - m.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>}
      </svg>
      <div className="row" style={{ gap: '0.3rem 1rem', marginTop: '0.2rem' }}>
        {series.map((s) => <span key={s.name} className="small" style={{ color: 'var(--ink-2)' }}><span className="swatch" style={{ background: s.color }} />{s.name}</span>)}
      </div>
      {tip.node}
    </div>
  );
}

/** Heat-table for confusion matrices: rows = true class, cols = predicted. */
export function MatrixTable({ m, rowLabels, colLabels, normalize = 'row', format, color = '#0d1317', onCell, highlight }: {
  m: number[][]; rowLabels: string[]; colLabels: string[]; normalize?: 'row' | 'none'; format?: (v: number, raw: number) => string;
  color?: string; onCell?: (i: number, j: number) => void; highlight?: [number, number] | null;
}) {
  const tip = useTooltip();
  const rows = m.map((r) => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => (normalize === 'row' ? v / s : v)); });
  const mx = Math.max(...rows.flat(), 1e-9);
  const rgb = d3.color(color)!.rgb();
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ width: 'auto', borderCollapse: 'separate', borderSpacing: 2 }}>
        <thead>
          <tr><th style={{ border: 0 }}><span className="muted small">true ↓ · predicted →</span></th>{colLabels.map((c) => <th key={c} className="num" style={{ border: 0, fontSize: '0.75rem' }}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <th style={{ border: 0, fontSize: '0.8rem' }}>{rowLabels[i]}</th>
              {r.map((v, j) => {
                const a = v / mx;
                const isH = highlight && highlight[0] === i && highlight[1] === j;
                return (
                  <td key={j} className="num"
                    onClick={onCell ? () => onCell(i, j) : undefined}
                    onMouseMove={(e) => tip.show(e, <>true <b>{rowLabels[i]}</b> → predicted <b>{colLabels[j]}</b><br />{normalize === 'row' ? `${(v * 100).toFixed(1)}% of row (${m[i][j].toFixed(0)} images)` : m[i][j].toFixed(1)}</>)}
                    onMouseLeave={tip.hide}
                    style={{ minWidth: '3.3rem', border: isH ? '2px solid var(--focus)' : 0, cursor: onCell ? 'pointer' : 'default', borderRadius: 3,
                      background: `rgba(${rgb.r},${rgb.g},${rgb.b},${0.06 + 0.88 * a})`, color: a > 0.5 ? '#fff' : 'var(--ink)' }}>
                    {format ? format(v, m[i][j]) : normalize === 'row' ? (v * 100).toFixed(0) : m[i][j].toFixed(0)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {tip.node}
    </div>
  );
}
