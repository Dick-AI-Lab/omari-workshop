import * as d3 from 'd3';
import { useState } from 'react';
import { OverlapNote } from '../components/Disclosure';
import { Empty, FileError, Figure, MODEL_META, Seg, TechNotes, useTooltip } from '../components/ui';
import { useArtifacts } from '../data/artifacts';
import { MODEL_IDS, type Grid, type LabelEfficiency, type ModelId } from '../data/types';
import { finite, mean, sd } from '../lib/stats';

type Metric = 'macro_f1' | 'accuracy';

export interface LePoint { fraction: number; values: (number | null)[]; n: number; mean: number | null; sd: number | null; runs?: (string | null)[] }

/** Per fraction: the finite per-seed values, their count, mean and sample sd (ddof = 1). Nulls are "not run", never 0. */
export function summarizeGrid(le: LabelEfficiency, m: ModelId, metric: Metric): LePoint[] {
  const g: Grid | undefined = le.per_model[m]?.[metric];
  if (!g) return [];
  return le.fractions.map((f, i) => {
    const row = g[i] ?? [];
    const v = row.filter(finite);
    return { fraction: f, values: row, n: v.length, mean: mean(row), sd: sd(row), runs: le.per_model[m]?.runs?.[i] };
  });
}

const fmtFrac = (f: number) => (f < 0.1 ? `${+(f * 100).toFixed(1)}%` : `${Math.round(f * 100)}%`);

export function LabelEfficiencyFigure() {
  const { labelEfficiency: le, files } = useArtifacts();
  const [metric, setMetric] = useState<Metric>('macro_f1');
  const [shown, setShown] = useState<ModelId[]>(MODEL_IDS);
  const tip = useTooltip();
  const errored = files.labelEfficiency?.state === 'error';
  if (!le) {
    return (
      <Figure title="How many labels does each representation need?" prov={errored ? 'error' : 'missing'}
        caption="Produced from the label-fraction fine-tuning runs (fractions 1%–100%, several seeds per fraction).">
        {errored ? <FileError message={files.labelEfficiency!.message} /> : <Empty title="No label-efficiency runs exported">Run the label-fraction sweeps in notebooks 02–04, then build the bundle.</Empty>}
      </Figure>
    );
  }
  const hasAcc = MODEL_IDS.some((m) => le.per_model[m]?.accuracy);
  const met: Metric = metric === 'accuracy' && !hasAcc ? 'macro_f1' : metric;
  const present = MODEL_IDS.filter((m) => le.per_model[m]);
  const data = present.map((m) => ({ m, pts: summarizeGrid(le, m, met) }));
  const visible = data.filter((d) => shown.includes(d.m));
  const allVals = visible.flatMap((d) => d.pts.flatMap((p) => p.values.filter(finite)));
  const W = 640, H = 300, mg = { t: 14, r: 16, b: 40, l: 52 };
  const fr = le.fractions.filter((f) => f > 0);
  const x = (fr.length > 1 ? d3.scaleLog().domain([d3.min(fr)! / 1.4, d3.max(fr)! * 1.25]) : d3.scaleLinear().domain([0, 1.1])).range([mg.l, W - mg.r]);
  const lo = allVals.length ? Math.min(...allVals) : 0, hi = allVals.length ? Math.max(...allVals) : 1;
  const y = d3.scaleLinear().domain([Math.max(0, lo - 0.03), Math.min(1, hi + 0.02)]).nice().range([H - mg.b, mg.t]);
  const offset = (mi: number) => (mi - (present.length - 1) / 2) * 5; // small horizontal dodge per model, in px
  const regimeText = le.regime === 'finetune' ? 'full fine-tuning of every weight' : le.regime === 'frozen' ? 'a frozen encoder with a trained linear head' : le.regime;
  const singleSeed = data.flatMap((d) => d.pts.filter((p) => p.n === 1).map((p) => ({ m: d.m, f: p.fraction })));
  const notRun = data.filter((d) => d.pts.some((p) => p.n === 0));
  const metricName = met === 'macro_f1' ? 'macro-F1' : 'accuracy';

  return (
    <Figure title="How many labels does each representation need?" prov="model" provDetail={`${le.provenance.script} · regime: ${le.regime}`}
      caption={<>Test {metricName} after {regimeText}, using a fraction of the training labels sampled <b>by patient</b> and stratified by class (regime recorded in the bundle: <code>{le.regime}</code>). Small dots are individual seeds; the line and bars show the mean ± one sample standard deviation (ddof = 1) where at least two seeds exist. A point with a single seed is drawn hollow and has no bar. Missing runs are reported as not run, never interpolated or set to zero.</>}>
      <div className="row" style={{ marginBottom: '0.5rem' }}>
        <Seg label="Metric" value={met} onChange={setMetric} options={[{ value: 'macro_f1', label: 'Macro-F1' }, { value: 'accuracy', label: 'Accuracy', disabled: !hasAcc, title: hasAcc ? undefined : 'Accuracy grid not in this bundle (schema 1.0)' }]} />
        <div className="control">
          <span className="control-label"><span>Models</span></span>
          <div className="seg" role="group" aria-label="Models shown">
            {present.map((m) => <button key={m} type="button" aria-pressed={shown.includes(m)} onClick={() => setShown(shown.includes(m) ? shown.filter((x) => x !== m) : [...shown, m])}><span className="swatch" style={{ background: MODEL_META[m].color }} />{MODEL_META[m].name}</button>)}
          </div>
        </div>
      </div>
      <div style={{ position: 'relative' }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${metricName} versus fraction of training labels`}>
          <g className="grid">{y.ticks(5).map((t) => <line key={t} x1={mg.l} x2={W - mg.r} y1={y(t)} y2={y(t)} />)}</g>
          {visible.map(({ m, pts }) => {
            const mi = present.indexOf(m);
            const ok = pts.filter((p) => p.mean != null);
            const line = d3.line<LePoint>().x((p) => x(p.fraction) + offset(mi)).y((p) => y(p.mean!))(ok) ?? '';
            return (
              <g key={m}>
                {ok.length > 1 && <path d={line} fill="none" stroke={MODEL_META[m].color} strokeWidth={2} />}
                {ok.length === 1 && le.fractions.length > 1 && (
                  // bottom-right of the plot area: the curves converge near the top right, so a label next to the point overlaps them
                  <text x={W - mg.r - 4} y={H - mg.b - 10 - mi * 13} textAnchor="end" style={{ fill: MODEL_META[m].text, fontSize: 10.5 }}>{MODEL_META[m].name}: {fmtFrac(ok[0].fraction)} only, low-label sweep not run</text>
                )}
                {pts.map((p) => {
                  const cx = x(p.fraction) + offset(mi);
                  return (
                    <g key={p.fraction}>
                      {p.sd != null && <line x1={cx} x2={cx} y1={y(p.mean! - p.sd)} y2={y(p.mean! + p.sd)} stroke={MODEL_META[m].color} strokeWidth={2} />}
                      {p.values.map((v, si) => finite(v) && <circle key={si} cx={cx + (si - 1) * 2.2} cy={y(v)} r={2} fill={MODEL_META[m].color} opacity={0.55} />)}
                      {p.mean != null && (
                        <circle cx={cx} cy={y(p.mean)} r={4.5} fill={p.n === 1 ? 'var(--paper-raised)' : MODEL_META[m].color} stroke={MODEL_META[m].color} strokeWidth={2}
                          onMouseMove={(e) => tip.show(e, <>{MODEL_META[m].name} · {fmtFrac(p.fraction)} labels<br />{metricName}: {p.mean!.toFixed(3)}{p.sd != null ? ` ± ${p.sd.toFixed(3)} (sd)` : ''}<br />seeds run: {p.n} of {p.values.length}<br />{p.values.map((v) => (finite(v) ? v.toFixed(3) : 'not run')).join(' · ')}</>)}
                          onMouseLeave={tip.hide} />
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}
          <g className="axis">
            <line x1={mg.l} x2={W - mg.r} y1={H - mg.b} y2={H - mg.b} />
            {le.fractions.map((f) => <text key={f} x={x(f)} y={H - mg.b + 15} textAnchor="middle">{fmtFrac(f)}</text>)}
            {y.ticks(5).map((t) => <text key={t} x={mg.l - 6} y={y(t) + 4} textAnchor="end">{t.toFixed(2)}</text>)}
          </g>
          <text className="axis-title" x={(mg.l + W - mg.r) / 2} y={H - 4} textAnchor="middle">fraction of training labels (log scale)</text>
          <text className="axis-title" transform={`translate(12, ${(mg.t + H - mg.b) / 2}) rotate(-90)`} textAnchor="middle">test {metricName}</text>
        </svg>
        {tip.node}
      </div>
      <div style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table className="data small">
          <thead><tr><th>Labels</th>{present.map((m) => <th key={m} className="num" style={{ color: MODEL_META[m].text }}>{MODEL_META[m].name}</th>)}</tr></thead>
          <tbody>{le.fractions.map((f, i) => (
            <tr key={f}><td>{fmtFrac(f)}</td>{data.map(({ m, pts }) => { const p = pts[i]; return (
              <td key={m} className="num">{p && p.mean != null ? <>{p.mean.toFixed(3)}{p.sd != null ? ` ± ${p.sd.toFixed(3)}` : ''} <span className="muted">(n={p.n})</span></> : <span className="notrun">not run</span>}</td>
            ); })}</tr>
          ))}</tbody>
        </table>
      </div>
      <ul className="small" style={{ fontFamily: 'var(--sans)', marginBottom: 0 }}>
        {notRun.map(({ m, pts }) => (
          <li key={m}><b style={{ color: MODEL_META[m].text }}>{MODEL_META[m].name}</b>: {le.availability?.[m] ?? `not run at ${pts.filter((p) => p.n === 0).map((p) => fmtFrac(p.fraction)).join(', ')}.`}{!le.availability?.[m] && pts.filter((p) => p.n > 0).length === 1 ? ' Low-label sweep not run; shown at a single fraction only.' : ''}</li>
        ))}
        {singleSeed.length > 0 && <li>Single-seed points (no spread available): {singleSeed.map((s) => `${MODEL_META[s.m].name} at ${fmtFrac(s.f)}`).join(', ')}.</li>}
      </ul>
      <TechNotes notes={le.notes} />
      <OverlapNote context="read the low-label points with this in mind">{' '}Any USF-MAE advantage at small label fractions is therefore suggestive, not proof that ultrasound self-supervision is better: architecture, pretraining data and fine-tuning recipe all differ, seeds are few, and the pretraining corpus contained these images.</OverlapNote>
    </Figure>
  );
}
