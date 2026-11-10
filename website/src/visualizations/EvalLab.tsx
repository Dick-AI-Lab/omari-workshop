import { useEffect, useMemo, useState } from 'react';
import { LineChart, MatrixTable } from '../components/charts';
import { Empty, Figure, GrayCanvas, MODEL_META, Prov, Seg, Slider, pct } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { CLASS_SHORT, FETAL_PLANES_LITERATURE as LIT } from '../data/literature';
import { MODEL_IDS, type ModelId } from '../data/types';
import { Gray } from '../lib/imageOps';
import { reliability, rocCurve, shiftPrevalence, softmax, summary, nll } from '../lib/metrics';
import { gaussian, mulberry32 } from '../lib/rng';

function useEvalModels() {
  const { evaluation } = useArtifacts();
  return MODEL_IDS.filter((m) => evaluation?.per_model[m]);
}

export function EvaluationSummary() {
  const { evaluation } = useArtifacts();
  const models = useEvalModels();
  const [model, setModel] = useState<ModelId>('densenet161');
  const [cell, setCell] = useState<[number, number] | null>(null);
  const [metric, setMetric] = useState<'f1' | 'precision' | 'recall'>('f1');
  const m = models.includes(model) ? model : models[0];
  if (!evaluation || !m) {
    return (
      <Figure title="Test-set performance" prov="missing" caption="Confusion matrices, per-class metrics and confidence intervals are read from evaluation.json, written by notebook 05 from saved test-set predictions.">
        <Empty title="No evaluation results exported">
          Train with notebooks 02–04, then run notebook 05 (or <code>python scripts/evaluate.py --config configs/export.yaml</code>). No numbers are shown until then. The DenseNet-169 figures in the original paper (6.2% top-1 error) used a different model and protocol, so they are not a stand-in.
        </Empty>
      </Figure>
    );
  }
  const pm = evaluation.per_model[m]!;
  const tp = evaluation.test_predictions;
  const cellItems = cell && tp ? tp.ids.map((id, i) => ({ id, i })).filter(({ i }) => {
    const p = tp.probs[m]?.[i];
    if (!p) return false;
    const pred = p.indexOf(Math.max(...p));
    return tp.labels[i] === cell[0] && pred === cell[1];
  }) : [];
  const misc = evaluation.misclassified ?? [];
  return (
    <Figure title="Test-set performance" prov="model" provDetail={`${evaluation.provenance.script} · ${evaluation.provenance.generated_at}`}
      caption={<>{evaluation.split} split, n = {evaluation.n}. Rows are expert labels, columns are predictions, cells are row percentages (recall on the diagonal). Click a cell to list its images. 95% intervals are percentile bootstrap over test images{evaluation.comparisons?.length ? '; pairwise tests are McNemar on the same images' : ''}. Because test images cluster within patients, image-level bootstrap intervals are optimistic.</>}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <Seg label="Model" value={m} onChange={(v) => { setModel(v); setCell(null); }} options={models.map((id) => ({ value: id, label: MODEL_META[id].name }))} />
      </div>
      <div className="split">
        <MatrixTable m={pm.confusion} rowLabels={CLASS_SHORT} colLabels={CLASS_SHORT} color={MODEL_META[m].color === 'var(--m-vit)' ? '#b07800' : MODEL_META[m].color === 'var(--m-usfmae)' ? '#00775a' : '#0072b2'}
          onCell={(i, j) => setCell([i, j])} highlight={cell} />
        <div>
          <table className="data">
            <thead><tr><th></th>{models.map((id) => <th key={id} className="num" style={{ color: MODEL_META[id].text }}>{MODEL_META[id].name}</th>)}</tr></thead>
            <tbody>
              {([['Accuracy', 'accuracy'], ['Macro-F1', 'macro_f1'], ['Balanced acc.', 'balanced_accuracy']] as const).map(([lab, key]) => (
                <tr key={key}><td>{lab}</td>{models.map((id) => {
                  const e = evaluation.per_model[id]!;
                  const ci = key === 'accuracy' ? e.ci95?.accuracy : key === 'macro_f1' ? e.ci95?.macro_f1 : undefined;
                  return <td key={id} className="num">{pct(e[key])}{ci && <div className="muted small">[{pct(ci[0])}, {pct(ci[1])}]</div>}</td>;
                })}</tr>
              ))}
              <tr><td>ECE (10 bins)</td>{models.map((id) => <td key={id} className="num">{evaluation.per_model[id]!.ece != null ? pct(evaluation.per_model[id]!.ece!) : '—'}</td>)}</tr>
            </tbody>
          </table>
          {evaluation.comparisons?.map((c) => <p key={c.a + c.b} className="small muted" style={{ margin: '0.3rem 0' }}>{MODEL_META[c.a].name} vs {MODEL_META[c.b].name}: {c.test}, statistic {c.statistic.toFixed(2)}, p = {c.p_value < 0.001 ? '<0.001' : c.p_value.toFixed(3)}{c.note ? ` (${c.note})` : ''}</p>)}
          <div style={{ marginTop: '0.8rem' }}>
            <Seg label="Per-class metric" value={metric} onChange={setMetric} options={[{ value: 'f1', label: 'F1' }, { value: 'precision', label: 'Precision' }, { value: 'recall', label: 'Recall' }]} />
            <PerClassGrouped models={models} metric={metric} />
          </div>
        </div>
      </div>
      {cell && (
        <div style={{ marginTop: '1rem' }}>
          <b className="small">True {CLASS_SHORT[cell[0]]} → predicted {CLASS_SHORT[cell[1]]}: {cellItems.length} images</b>
          <Gallery ids={cellItems.map((c) => c.id)} misc={misc} />
        </div>
      )}
    </Figure>
  );
}

function PerClassGrouped({ models, metric }: { models: ModelId[]; metric: 'f1' | 'precision' | 'recall' }) {
  const { evaluation } = useArtifacts();
  const W = 520, rowH = 34, lw = 70;
  return (
    <svg viewBox={`0 0 ${W} ${CLASS_SHORT.length * rowH + 10}`} width="100%" role="img" aria-label={`per-class ${metric}`}>
      {CLASS_SHORT.map((c, ci) => (
        <g key={c} transform={`translate(0, ${ci * rowH + 4})`}>
          <text x={lw - 6} y={rowH / 2 + 3} textAnchor="end" style={{ fill: 'var(--ink)' }}>{c}</text>
          {models.map((m, mi) => {
            const v = evaluation!.per_model[m]!.per_class[ci][metric];
            const bh = (rowH - 8) / models.length;
            return <g key={m}><rect x={lw} y={mi * bh + 2} width={(W - lw - 50) * v} height={bh - 1} fill={MODEL_META[m].color} rx={1} />
              <text x={lw + (W - lw - 50) * v + 4} y={mi * bh + bh / 2 + 5} style={{ fontSize: 9.5 }}>{(v * 100).toFixed(1)}</text></g>;
          })}
        </g>
      ))}
    </svg>
  );
}

function Gallery({ ids, misc }: { ids: string[]; misc: { id: string; image: string }[] }) {
  const { manifest } = useArtifacts();
  const lookup = new Map<string, string>();
  manifest?.samples.forEach((s) => lookup.set(s.id, s.image));
  misc.forEach((s) => lookup.set(s.id, s.image));
  const withImg = ids.filter((id) => lookup.has(id)).slice(0, 12);
  const [imgs, setImgs] = useState<Gray[]>([]);
  useEffect(() => { let c = false; Promise.all(withImg.map((id) => loadGrayPNG(lookup.get(id)!))).then((g) => !c && setImgs(g)).catch(() => {}); return () => { c = true; }; }, [withImg.join(',')]); // eslint-disable-line
  if (!ids.length) return <p className="small muted">No images in this cell.</p>;
  return (
    <>
      <div className="grid-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(8rem, 1fr))', marginTop: '0.4rem' }}>
        {imgs.map((g, i) => <GrayCanvas key={withImg[i]} img={g} label={withImg[i]} />)}
      </div>
      {ids.length > withImg.length && <p className="small muted">{ids.length - withImg.length} further images in this cell were not exported as thumbnails (exporter keeps up to <code>max_misclassified</code> per cell).</p>}
    </>
  );
}

/** Hypothetical confusion matrix for when no model has been evaluated — clearly illustrative. */
const HYPOTHETICAL: number[][] = [
  [86, 0, 2, 8, 0, 4], [0, 97, 0, 0, 0, 3], [1, 0, 90, 1, 0, 8], [5, 0, 1, 88, 0, 6], [0, 0, 0, 0, 98, 2], [2, 2, 4, 3, 1, 88],
];

export function PrevalenceSimulator() {
  const { evaluation } = useArtifacts();
  const models = useEvalModels();
  const [model, setModel] = useState<ModelId>(models[0] ?? 'densenet161');
  const real = evaluation?.per_model[model]?.confusion;
  const cm = real ?? HYPOTHETICAL;
  const base = LIT.classCounts.map((c) => c / LIT.totalImages);
  const [priors, setPriors] = useState<number[]>(base);
  const [focus, setFocus] = useState(0);
  const shifted = useMemo(() => shiftPrevalence(cm, priors), [cm, priors]);
  const sBase = summary(shiftPrevalence(cm, base));
  const s = summary(shifted);
  const presets: Record<string, number[]> = {
    'Dataset mix': base,
    Uniform: base.map(() => 1 / 6),
    'Cervical-length clinic': [0.05, 0.05, 0.05, 0.05, 0.7, 0.1],
    'Growth scan (biometry)': [0.25, 0.25, 0.25, 0.05, 0.02, 0.18],
  };
  return (
    <Figure title="Same classifier, different clinic" prov={real ? ['model', 'live'] : ['illustrative', 'live']}
      caption={<>Holding each class’s behaviour fixed (the rows of the confusion matrix), change the deployment prevalences. Recall per class cannot change, but <b>precision does</b>, and so does overall accuracy. This is the arithmetic of prior shift and assumes the images of each class look the same in the new setting. {real ? `Using ${MODEL_META[model].name}'s test confusion matrix.` : <b>Using a hypothetical classifier (illustrative) because no evaluation has been exported.</b>}</>}>
      <div className="split">
        <div className="controls">
          {real && <Seg label="Model" value={model} onChange={setModel} options={models.map((id) => ({ value: id, label: MODEL_META[id].name }))} />}
          <Seg label="Scenario" value={Object.keys(presets).find((k) => presets[k].every((v, i) => Math.abs(v - priors[i]) < 1e-9)) ?? 'custom'} onChange={(k) => presets[k] && setPriors(presets[k])}
            options={[...Object.keys(presets).map((k) => ({ value: k, label: k })), { value: 'custom', label: 'Custom', disabled: true }]} />
          {CLASS_SHORT.map((c, i) => (
            <Slider key={c} label={`${c} prevalence`} value={priors[i]} min={0.005} max={0.8} step={0.005} onChange={(v) => setPriors(priors.map((p, j) => (j === i ? v : p)))}
              format={() => pct(priors[i] / priors.reduce((a, b) => a + b, 0), 1)} />
          ))}
        </div>
        <div>
          <Seg label="Class to inspect" value={focus} onChange={setFocus} options={CLASS_SHORT.map((c, i) => ({ value: i, label: c }))} />
          <table className="data" style={{ marginTop: '0.6rem' }}>
            <thead><tr><th></th><th className="num">Dataset mix</th><th className="num">This scenario</th></tr></thead>
            <tbody>
              <tr><td>{CLASS_SHORT[focus]} recall</td><td className="num">{pct(sBase.perClass[focus].recall)}</td><td className="num">{pct(s.perClass[focus].recall)}</td></tr>
              <tr><td>{CLASS_SHORT[focus]} precision (PPV)</td><td className="num">{pct(sBase.perClass[focus].precision)}</td><td className="num"><b>{pct(s.perClass[focus].precision)}</b></td></tr>
              <tr><td>Overall accuracy</td><td className="num">{pct(sBase.accuracy)}</td><td className="num"><b>{pct(s.accuracy)}</b></td></tr>
              <tr><td>Macro-F1</td><td className="num">{pct(sBase.macroF1)}</td><td className="num">{pct(s.macroF1)}</td></tr>
            </tbody>
          </table>
          <p className="small muted">Precision for class c: <i>π<sub>c</sub>·R<sub>c</sub> / Σ<sub>k</sub> π<sub>k</sub>·P(ŷ=c | y=k)</i>. A rare class with good recall can still be mostly false positives.</p>
        </div>
      </div>
    </Figure>
  );
}

function illustrativeProbs(n = 1500, seed = 5): { probs: number[][]; labels: number[] } {
  const rand = mulberry32(seed);
  const probs: number[][] = [], labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const y = Math.floor(rand() * 6);
    const z = Array.from({ length: 6 }, (_, c) => (c === y ? 1.6 : 0) + gaussian(rand) * 1.1);
    probs.push(softmax(z, 0.45)); // T < 1 → overconfident by construction
    labels.push(y);
  }
  return { probs, labels };
}

export function CalibrationLab() {
  const { evaluation } = useArtifacts();
  const models = useEvalModels();
  const tp = evaluation?.test_predictions;
  const [model, setModel] = useState<ModelId>(models[0] ?? 'densenet161');
  const [T, setT] = useState(1);
  const realProbs = tp?.probs[model];
  const data = useMemo(() => (realProbs && tp ? { probs: realProbs, labels: tp.labels } : illustrativeProbs()), [realProbs, tp]);
  // temperature what-if on log-probabilities (valid: logits are defined up to an additive constant)
  const scaled = useMemo(() => data.probs.map((p) => softmax(p.map((v) => Math.log(Math.max(v, 1e-12))), T)), [data, T]);
  const rel = useMemo(() => reliability(scaled, data.labels, 10), [scaled, data]);
  const rel1 = useMemo(() => reliability(data.probs, data.labels, 10), [data]);
  const bins = rel.bins.filter((b) => b.count > 0);
  return (
    <Figure title="Is 95% confident 95% correct?" prov={realProbs ? ['model', 'live'] : ['illustrative', 'live']}
      caption={<>Reliability diagram: test predictions are grouped by top-class confidence. A calibrated model lies on the diagonal. Expected calibration error (ECE) is the bin-weighted gap. The temperature slider rescales log-probabilities, <i>p</i> ∝ exp(log <i>p</i> / T). In practice <b>T must be fitted on the validation set</b> (notebook 05 does this); adjusting it on the test set, as here, is for intuition only. {realProbs ? '' : <b>No predictions exported: showing a synthetic over-confident classifier (illustrative).</b>}</>}>
      <div className="split">
        <div>
          <LineChart width={460} height={300} xLabel="confidence" yLabel="accuracy" yDomain={[0, 1]} xDomain={[0, 1]} diagonal
            series={[{ name: `T = ${T.toFixed(2)}`, color: realProbs ? MODEL_META[model].color : 'var(--p-illus)', x: bins.map((b) => b.conf), y: bins.map((b) => b.acc) }]}
            xTickFormat={(v) => v.toFixed(1)} yTickFormat={(v) => v.toFixed(1)} />
        </div>
        <div className="controls">
          {realProbs && <Seg label="Model" value={model} onChange={setModel} options={models.map((id) => ({ value: id, label: MODEL_META[id].name }))} />}
          <Slider label="Temperature T (what-if)" value={T} min={0.3} max={4} step={0.05} onChange={setT} format={(v) => v.toFixed(2)} />
          <table className="data">
            <tbody>
              <tr><td>ECE at T = 1</td><td className="num">{pct(rel1.ece)}</td></tr>
              <tr><td>ECE at T = {T.toFixed(2)}</td><td className="num"><b>{pct(rel.ece)}</b></td></tr>
              <tr><td>NLL at T = {T.toFixed(2)}</td><td className="num">{nll(scaled, data.labels).toFixed(3)}</td></tr>
              <tr><td>Accuracy (unchanged by T)</td><td className="num">{pct(summary(confusionFrom(scaled, data.labels)).accuracy)}</td></tr>
            </tbody>
          </table>
          <p className="small muted">Temperature scaling never changes the predicted class, only confidence. ECE with equal-width bins is sensitive to the number of bins and to sparse high-confidence bins.</p>
        </div>
      </div>
    </Figure>
  );
}

function confusionFrom(probs: number[][], labels: number[]): number[][] {
  const cm = Array.from({ length: 6 }, () => new Array(6).fill(0));
  probs.forEach((p, i) => { cm[labels[i]][p.indexOf(Math.max(...p))]++; });
  return cm;
}

export function ThresholdLab() {
  const { evaluation } = useArtifacts();
  const models = useEvalModels();
  const tp = evaluation?.test_predictions;
  const [cls, setCls] = useState(1);
  const [thr, setThr] = useState(0.5);
  if (!tp || !models.length) {
    return (
      <Figure title="One-vs-rest operating points" prov="missing" caption="ROC curves and threshold trade-offs need per-image test probabilities (evaluation.json → test_predictions).">
        <Empty title="No per-image predictions exported">Enable <code>export_test_probabilities: true</code> in <code>configs/export.yaml</code>.</Empty>
      </Figure>
    );
  }
  const curves = models.map((m) => {
    const scores = tp.probs[m]!.map((p) => p[cls]);
    const pos = tp.labels.map((y) => y === cls);
    const roc = rocCurve(scores, pos);
    const P = pos.filter(Boolean).length, N = pos.length - P;
    const tpN = scores.filter((s, i) => s >= thr && pos[i]).length, fpN = scores.filter((s, i) => s >= thr && !pos[i]).length;
    return { m, roc, tpr: P ? tpN / P : 0, fpr: N ? fpN / N : 0, ppv: tpN + fpN ? tpN / (tpN + fpN) : 0 };
  });
  return (
    <Figure title="One-vs-rest operating points" prov={['model', 'live']}
      caption={<>The six-way decision is an argmax and has no single threshold. A threshold <i>is</i> meaningful for a one-vs-rest question such as “is this a usable brain plane?”. Score = predicted probability of the chosen class. Changing the threshold trades sensitivity against false positives; the ROC curve does not depend on prevalence, but the positive predictive value does.</>}>
      <div className="split">
        <LineChart width={460} height={300} xLabel="false-positive rate" yLabel="true-positive rate" yDomain={[0, 1]} xDomain={[0, 1]} diagonal
          series={curves.map((c) => ({ name: `${MODEL_META[c.m].name} (AUC ${c.roc.auc.toFixed(3)})`, color: MODEL_META[c.m].color, x: thin(c.roc.fpr), y: thin(c.roc.tpr) }))}
          xTickFormat={(v) => v.toFixed(1)} yTickFormat={(v) => v.toFixed(1)} />
        <div className="controls">
          <Seg label="Positive class" value={cls} onChange={setCls} options={CLASS_SHORT.map((c, i) => ({ value: i, label: c }))} />
          <Slider label="Threshold on p(class)" value={thr} min={0.01} max={0.99} step={0.01} onChange={setThr} format={(v) => v.toFixed(2)} />
          <table className="data">
            <thead><tr><th></th><th className="num">Sensitivity</th><th className="num">FPR</th><th className="num">PPV</th></tr></thead>
            <tbody>{curves.map((c) => <tr key={c.m}><td style={{ color: MODEL_META[c.m].text }}>{MODEL_META[c.m].name}</td><td className="num">{pct(c.tpr)}</td><td className="num">{pct(c.fpr, 2)}</td><td className="num">{pct(c.ppv)}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
    </Figure>
  );
}

const thin = (a: number[], n = 200) => (a.length <= n ? a : a.filter((_, i) => i % Math.ceil(a.length / n) === 0 || i === a.length - 1));

export function LearningCurves() {
  const { labelEfficiency: le, curves } = useArtifacts();
  const [regime, setRegime] = useState<string>('');
  const [model, setModel] = useState<ModelId>('densenet161');
  const regimes = curves ? Array.from(new Set(Object.values(curves.per_model).flatMap((r) => Object.keys(r ?? {})))) : [];
  const reg = regime || regimes[0];
  const cModels = MODEL_IDS.filter((m) => curves?.per_model[m]?.[reg]);
  const cm = cModels.includes(model) ? model : cModels[0];
  return (
    <>
      <Figure title="How many labels does each representation need?" prov={le ? 'model' : 'missing'} provDetail={le ? `${le.provenance.script} · regime: ${le.regime}` : undefined}
        caption={le ? <>Test macro-F1 when fine-tuning on a fraction of the training labels, sampled <b>by patient</b> and stratified by class (regime: {le.regime}). Lines are means over {le.seeds.length} seed{le.seeds.length === 1 ? '' : 's'}; bands show min–max. This is where differences in pretraining are expected to matter most.</> : 'Produced by notebook 05 from label-fraction runs of notebooks 02–04 (fractions 1%–100%, several seeds).'}>
        {le ? (
          <LineChart xLog xLabel="fraction of training labels" yLabel="macro-F1" xTickFormat={(v) => pct(v, v < 0.1 ? 0 : 0)} yTickFormat={(v) => v.toFixed(2)}
            series={MODEL_IDS.filter((m) => le.per_model[m]).map((m) => {
              const rows = le.per_model[m]!.macro_f1;
              return { name: MODEL_META[m].name, color: MODEL_META[m].color, x: le.fractions, y: rows.map((r) => r.reduce((a, b) => a + b, 0) / r.length), band: [rows.map((r) => Math.min(...r)), rows.map((r) => Math.max(...r))] };
            })} />
        ) : <Empty title="No label-efficiency runs exported">Run notebooks 02–04 with <code>LABEL_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]</code>, then notebook 05.</Empty>}
      </Figure>
      <Figure title="Training and validation curves" prov={curves ? 'model' : 'missing'}
        caption="Loss on training and patient-disjoint validation data, with validation macro-F1 (dashed) on the same axis. Diverging train and validation loss marks the onset of overfitting; the checkpoint kept is the one with the best validation macro-F1.">
        {curves && cm ? <>
          <div className="row">
            <Seg label="Model" value={cm} onChange={setModel} options={cModels.map((id) => ({ value: id, label: MODEL_META[id].name }))} />
            <Seg label="Regime" value={reg} onChange={setRegime} options={regimes.map((r) => ({ value: r, label: r }))} />
          </div>
          {(() => { const c = curves.per_model[cm]![reg]; return (
            <LineChart xLabel="epoch" yLabel="loss / macro-F1" series={[
              { name: 'train loss', color: MODEL_META[cm].color, x: c.epoch, y: c.train_loss },
              { name: 'validation loss', color: 'var(--ink)', x: c.epoch, y: c.val_loss },
              { name: 'validation macro-F1', color: 'var(--p-model)', x: c.epoch, y: c.val_macro_f1, dashed: true },
            ]} />); })()}
        </> : <Empty title="No training curves exported">Notebooks 02–04 write <code>history.json</code> per run; notebook 05 collects them into <code>training_curves.json</code>.</Empty>}
      </Figure>
    </>
  );
}

export function DomainShiftTable() {
  const { evaluation } = useArtifacts();
  const models = useEvalModels();
  const machines = Array.from(new Set(models.flatMap((m) => Object.keys(evaluation?.per_model[m]?.by_machine ?? {}))));
  if (!machines.length) {
    return (
      <Figure title="Performance by ultrasound machine" prov="missing" caption="Acquisition-related shift: the same models evaluated separately on each machine recorded in the metadata.">
        <Empty title="Not exported">notebook 05 computes this when the metadata include <code>US_Machine</code>.</Empty>
      </Figure>
    );
  }
  return (
    <Figure title="Performance by ultrasound machine" prov="model"
      caption="Macro-F1 on test images from each machine. Subgroups are small and confounded with hospital, operator, period and case mix, so differences are hypotheses for shift, not proof of it. A stronger test trains on some machines and evaluates on a held-out one (leave-one-machine-out), which notebook 05 supports.">
      <table className="data">
        <thead><tr><th>Machine</th>{models.map((m) => <th key={m} className="num" style={{ color: MODEL_META[m].text }}>{MODEL_META[m].name}</th>)}<th className="num">n</th></tr></thead>
        <tbody>{machines.map((mc) => <tr key={mc}><td>{mc}</td>{models.map((m) => { const b = evaluation!.per_model[m]!.by_machine?.[mc]; return <td key={m} className="num">{b ? pct(b.macro_f1) : '—'}</td>; })}<td className="num">{evaluation!.per_model[models[0]]!.by_machine?.[mc]?.n ?? '—'}</td></tr>)}</tbody>
      </table>
    </Figure>
  );
}

export { Prov };
