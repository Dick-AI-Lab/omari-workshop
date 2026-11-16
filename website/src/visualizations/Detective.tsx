import { useEffect, useMemo, useState } from 'react';
import { ImagePicker } from '../components/ImagePicker';
import { Empty, FileError, Figure, GrayCanvas, MODEL_META, Slider, TechNotes, pct } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { layerLabel, methodLong, methodShort } from '../data/labels';
import { useImageSource } from '../data/imageSource';
import { CLASS_SHORT } from '../data/literature';
import { MODEL_IDS, type FaithfulnessRow, type ModelId, type PerturbationEntry, type SampleInfo } from '../data/types';
import { Gray, annotationMask, makeGray, resize } from '../lib/imageOps';
import { mulberry32 } from '../lib/rng';
import { toyAttribution, toyFeatureMaps, unitScale } from '../lib/toyModels';

const HYPOTHESES = [
  { id: 'overlay', text: 'At least one model depends on burned-in text, calipers or markers' },
  { id: 'contrast', text: 'Predictions change when contrast or gain changes' },
  { id: 'speckle', text: 'Added speckle flips at least one prediction' },
  { id: 'occlude', text: 'Masking the anatomy the map highlights lowers the true-class probability' },
  { id: 'stable', text: 'The explanation map stays similar under small perturbations' },
];

export function spearman(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const rank = (v: ArrayLike<number>) => {
    const idx = Array.from(v).map((x, i) => [x, i] as [number, number]).sort((p, q) => p[0] - q[0]);
    const r = new Array(idx.length);
    idx.forEach(([, i], k) => (r[i] = k));
    return r as number[];
  };
  const ra = rank(a), rb = rank(b), n = ra.length;
  const ma = (n - 1) / 2;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - ma); da += (ra[i] - ma) ** 2; db += (rb[i] - ma) ** 2; }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

function useGray(path?: string) {
  const [g, setG] = useState<Gray | null>(null);
  useEffect(() => { let c = false; setG(null); if (path) loadGrayPNG(path).then((x) => !c && setG(x)).catch((e) => console.error('[detective]', path, e)); return () => { c = true; }; }, [path]);
  return g;
}

function MapSimilarity({ a, b }: { a?: string; b?: string }) {
  const ga = useGray(a), gb = useGray(b);
  if (!a || !b) return <span className="notrun">n/a</span>;
  if (!ga || !gb) return <span className="muted">…</span>;
  const rb = gb.w === ga.w && gb.h === ga.h ? gb : resize(gb, ga.w, ga.h);
  return <span>{spearman(ga.data, rb.data).toFixed(2)}</span>;
}

/** Layer the exporter used for perturbation maps (recorded in 1.1; 1.0 default: DenseNet 'act', ViTs 'block11'). */
const defaultPertLayer = (m: ModelId) => (m === 'densenet161' ? 'act' : 'block11');

function baseMapFor(s: SampleInfo, m: ModelId, p: PerturbationEntry) {
  const pa = p.attributions?.[m];
  if (!pa) return undefined;
  const layer = pa.layer ?? defaultPertLayer(m);
  const target = pa.target_class ?? s.label;
  const c = s.attributions?.filter((a) => a.model === m && a.method === pa.method && a.target_class === target) ?? [];
  return c.find((a) => a.layer === layer) ?? c[0];
}

export function DetectiveLab() {
  const src = useImageSource();
  const [chosen, setChosen] = useState<string[]>([]);
  const [locked, setLocked] = useState(false);
  const [thresh, setThresh] = useState(0.92);
  const [detail, setDetail] = useState<number | null>(null);
  const s = src.sample;
  const perts = s?.perturbations ?? [];
  const mask = useMemo(() => annotationMask(src.gray, thresh, 1), [src.gray, thresh]);
  const maskImg = useMemo(() => { const g = makeGray(src.gray.w, src.gray.h); mask.forEach((m, i) => (g.data[i] = m)); return g; }, [mask, src.gray]);
  const frac = mask.reduce((a: number, b) => a + b, 0) / mask.length;
  const models = (Object.keys(MODEL_META) as ModelId[]).filter((m) => s?.predictions[m]);
  const sel = detail != null ? perts[detail] : undefined;
  // compare perturbations with predictions made by the same pipeline (exported PNG), when the bundle provides them
  const baseOf = (m: ModelId) => s?.predictions_display_png?.[m] ?? s?.predictions[m];

  return (
    <Figure title="Model detective" prov={perts.length ? ['live', 'model'] : ['live', 'missing']}
      caption={<>Commit to hypotheses <i>before</i> looking at the evidence; the reveal is locked until you do. The annotation detector (bright, saturated pixels) runs live on the selected image and is only a rough guide. Perturbation evidence comes only from controlled experiments run by the export pipeline on the real checkpoints: each row changes one factor and records every model’s prediction and attribution map for the expert label. Map ρ is the Spearman rank correlation between the original and the perturbed map of the same model and method, computed in your browser from the exported PNGs. Click a row to see the perturbed image and the maps.</>}>
      <ImagePicker compact />
      <div className="split">
        <div className="stack">
          <b className="small">1 · Your hypotheses for this image</b>
          {HYPOTHESES.map((h) => (
            <label key={h.id} className="row small" style={{ flexWrap: 'nowrap', alignItems: 'flex-start' }}>
              <input type="checkbox" disabled={locked} checked={chosen.includes(h.id)} onChange={(e) => setChosen(e.target.checked ? [...chosen, h.id] : chosen.filter((x) => x !== h.id))} /> {h.text}
            </label>
          ))}
          <div className="row">
            <button className="btn primary" disabled={!chosen.length || locked} onClick={() => setLocked(true)}>Lock hypotheses and reveal evidence</button>
            {locked && <button className="btn" onClick={() => { setLocked(false); setChosen([]); setDetail(null); }}>Start again</button>}
          </div>
          <b className="small" style={{ marginTop: '0.6rem' }}>2 · Where are the overlays? <span className="prov live" style={{ marginLeft: 4 }}>Live</span></b>
          <div className="grid-2">
            <GrayCanvas img={src.gray} overlay={{ heat: maskImg, opacity: 0.85, cmap: 'inferno', threshold: 0.5 }} label={`${pct(frac, 2)} of pixels flagged`} />
            <Slider label="Brightness threshold" value={thresh} min={0.7} max={0.99} step={0.01} onChange={setThresh} format={(v) => v.toFixed(2)} />
          </div>
        </div>
        <div className="stack">
          <b className="small">3 · Evidence <span className="prov model" style={{ marginLeft: 4 }}>Model output</span></b>
          {!locked ? <Empty title="Locked">Choose at least one hypothesis first.</Empty> :
            !perts.length ? (
              <Empty title={s ? 'No controlled perturbation experiments exported for this sample' : 'Select an exported sample'}>
                The pipeline runs, for each detective sample: contrast ×0.6/×1.5, brightness ±0.15, speckle 0.3/0.6, resolution ½ and ¼, annotation inpainting, and occlusion of the top-attributed region. Enable <code>perturbations: true</code> in <code>configs/export.yaml</code>.
              </Empty>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="data small">
                  <thead><tr><th>Change</th>{models.map((m) => <th key={m} style={{ color: MODEL_META[m].text }}>{MODEL_META[m].name}<br /><span className="muted">p(true) · pred · map ρ</span></th>)}</tr></thead>
                  <tbody>
                    <tr><td>none (original{s!.predictions_display_png ? ', exported PNG' : ''})</td>{models.map((m) => <td key={m}>{pct(baseOf(m)!.probs[s!.label])} · {CLASS_SHORT[baseOf(m)!.pred]} · 1.00</td>)}</tr>
                    {perts.map((p, pi) => (
                      <tr key={p.type + p.param} onClick={() => setDetail(detail === pi ? null : pi)} style={{ cursor: 'pointer', background: detail === pi ? '#eef2f4' : undefined }}>
                        <td>{p.label}</td>
                        {models.map((m) => {
                          const pr = p.predictions[m]; const b = baseOf(m)!;
                          if (!pr) return <td key={m}><span className="notrun">not run</span></td>;
                          const flip = pr.pred !== b.pred;
                          const d = pr.probs[s!.label] - b.probs[s!.label];
                          return <td key={m} style={{ color: flip ? 'var(--p-missing)' : undefined, fontWeight: flip ? 600 : 400 }}>{pct(pr.probs[s!.label])} ({d >= 0 ? '+' : ''}{(d * 100).toFixed(1)}) · {CLASS_SHORT[pr.pred]} · <MapSimilarity a={baseMapFor(s!, m, p)?.png} b={p.attributions?.[m]?.png} /></td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="small muted">Red = the predicted class changed. Maps: {models.map((m) => { const pa = perts[0]?.attributions?.[m]; return `${MODEL_META[m].name} ${pa ? `${pa.method_label ?? methodShort(pa.method)} (${layerLabel(m, pa.layer ?? defaultPertLayer(m), pa.layer_label)})` : 'not exported'}`; }).join('; ')}, all for the expert label. ρ near 1 means the map barely moved.{s?.predictions_display_png ? ' The unperturbed row uses predictions on the exported PNG, the same pipeline as the perturbed rows (it can differ slightly from the evaluation-run probabilities shown in Chapter 1).' : ''} Compare the results with the hypotheses you locked in, and check the faithfulness table below before trusting any one map.</p>
              </div>
            )}
        </div>
      </div>
      {locked && sel && s && <PerturbationDetail s={s} p={sel} models={models} />}
    </Figure>
  );
}

function PerturbationDetail({ s, p, models }: { s: SampleInfo; p: PerturbationEntry; models: ModelId[] }) {
  const orig = useGray(s.image);
  const pimg = useGray(p.image_png);
  return (
    <div style={{ marginTop: '1rem' }}>
      <b className="small">{p.label}: original versus perturbed input, and each model’s map before and after</b>
      <div className="grid-4" style={{ gridTemplateColumns: `repeat(${2 + models.length * 2}, minmax(0, 1fr))`, marginTop: '0.4rem' }}>
        {orig ? <GrayCanvas img={orig} label="original" /> : <div className="well" />}
        {pimg ? <GrayCanvas img={pimg} label="perturbed (exported)" /> : <div className="well" />}
        {models.flatMap((m) => [
          <MapTile key={m + 'a'} base={orig} path={baseMapFor(s, m, p)?.png} label={`${MODEL_META[m].name} before`} />,
          <MapTile key={m + 'b'} base={pimg} path={p.attributions?.[m]?.png} label={`${MODEL_META[m].name} after`} />,
        ])}
      </div>
    </div>
  );
}

function MapTile({ base, path, label }: { base: Gray | null; path?: string; label: string }) {
  const heat = useGray(path);
  if (!path) return <Empty title="not exported">{label}</Empty>;
  if (!base) return <div className="well" />;
  return <GrayCanvas img={base} overlay={heat ? { heat, opacity: 0.6, cmap: 'inferno' } : undefined} label={label} />;
}

/** Deletion / insertion AUC over a random subset of test images (faithfulness.json, schema 1.1). */
export function FaithfulnessTable() {
  const { faithfulness: f, files } = useArtifacts();
  if (!f) {
    const err = files.faithfulness?.state === 'error';
    return (
      <Figure title="Do the maps point at what the model uses? Deletion and insertion" prov={err ? 'error' : 'missing'}
        caption="Deletion: remove the most-attributed pixels first and watch the true-class probability fall (lower AUC = more faithful). Insertion: start from a blurred image and restore the most-attributed pixels first (higher AUC = more faithful).">
        {err ? <FileError message={files.faithfulness!.message} /> : <Empty title="Faithfulness results not in this bundle">Notebook 06 writes <code>faithfulness.csv</code>; the schema 1.1 bundle finaliser converts it to <code>faithfulness.json</code>.</Empty>}
      </Figure>
    );
  }
  const rb = f.random_baseline;
  const randomFor = (m: ModelId): Partial<FaithfulnessRow> | undefined => {
    const inRows = f.rows.find((r) => r.model === m && r.method === 'random');
    if (inRows) return inRows;
    if (Array.isArray(rb)) return rb.find((r) => r.model === m);
    return rb?.[m] ?? undefined;
  };
  const models = MODEL_IDS.filter((m) => f.rows.some((r) => r.model === m));
  const ms = (m: number | null | undefined, s: number | null | undefined) => (m == null ? '—' : `${m.toFixed(3)}${s != null ? ` ± ${s.toFixed(3)}` : ''}`);
  const notes = Array.isArray(f.notes) ? f.notes : f.notes ? [f.notes] : [];
  return (
    <Figure title="Do the maps point at what the model uses? Deletion and insertion" prov="model" provDetail={`${f.provenance.script} · ${f.provenance.generated_at}`}
      caption={<>Deletion and insertion AUC (Petsiuk et al., 2018) on {f.n_images ?? '?'} test images, {f.steps ?? '?'} steps, baseline: {f.baseline ?? 'not recorded'}, score: {f.target ?? 'not recorded'}. Values are mean ± sd across images. <b>Deletion</b>: lower is better (the probability collapses quickly when the top-ranked pixels go). <b>Insertion</b>: higher is better. Each model is compared with a random map on the same images; a method that does not beat random on a metric is flagged. With n = {f.n_images ?? '?'} and standard deviations this large, small differences are not significant (no paired test was run).</>}>
      <div style={{ overflowX: 'auto' }}>
        <table className="data small">
          <thead><tr><th>Model</th><th>Method</th><th>Layer</th><th className="num">Deletion AUC ↓</th><th className="num">Insertion AUC ↑</th><th>vs random</th></tr></thead>
          <tbody>
            {models.flatMap((m) => {
              const r0 = randomFor(m);
              const rows = f.rows.filter((r) => r.model === m && r.method !== 'random');
              return [
                ...rows.map((r) => {
                  const delWorse = r0?.deletion_auc_mean != null && r.deletion_auc_mean != null && r.deletion_auc_mean >= r0.deletion_auc_mean;
                  const insWorse = r0?.insertion_auc_mean != null && r.insertion_auc_mean != null && r.insertion_auc_mean <= r0.insertion_auc_mean;
                  return (
                    <tr key={m + r.method + (r.layer ?? '')}>
                      <td style={{ color: MODEL_META[m].text }}>{MODEL_META[m].name}</td>
                      <td title={methodLong(r.method)}>{r.method_label ?? methodShort(r.method)}</td>
                      <td>{r.layer_label ?? (r.layer ? layerLabel(m, r.layer) : <span className="muted">default</span>)}</td>
                      <td className="num" style={{ color: delWorse ? 'var(--p-missing)' : undefined }}>{ms(r.deletion_auc_mean, r.deletion_auc_sd)}</td>
                      <td className="num" style={{ color: insWorse ? 'var(--p-missing)' : undefined }}>{ms(r.insertion_auc_mean, r.insertion_auc_sd)}</td>
                      <td className="small">{!r0 ? '—' : delWorse && insWorse ? <b style={{ color: 'var(--p-missing)' }}>no better than random on either</b> : delWorse ? 'deletion not better' : insWorse ? 'insertion not better' : 'better on both'}</td>
                    </tr>
                  );
                }),
                r0 ? <tr key={m + 'random'} className="muted"><td>{MODEL_META[m].name}</td><td>Random map (baseline)</td><td>—</td><td className="num">{ms(r0.deletion_auc_mean, r0.deletion_auc_sd)}</td><td className="num">{ms(r0.insertion_auc_mean, r0.insertion_auc_sd)}</td><td /></tr> : null,
              ];
            })}
          </tbody>
        </table>
      </div>
      <p className="small" style={{ marginBottom: 0 }}>A better-than-random score says the highlighted pixels matter <i>to this model</i> under this removal procedure. It is <b>not proof of clinical reliability</b>, and it does not establish causality: blurring creates out-of-distribution inputs, and the ranking can change with the baseline, the number of steps and the image subset.</p>
      <TechNotes notes={notes} />
    </Figure>
  );
}

/** Adebayo-style cascading randomisation on the toy model: does the map depend on the weights at all? */
export function SanityCheckToy() {
  const src = useImageSource();
  const maps = useMemo(() => toyFeatureMaps(src.gray), [src.gray]);
  const trained = [1.2, -0.6, 0.8, 0.3, 0.2, 0];
  const [seed, setSeed] = useState(1);
  const randomW = useMemo(() => { const r = mulberry32(seed); return trained.map(() => r() * 2 - 0.5); }, [seed]); // eslint-disable-line
  const a = useMemo(() => unitScale(toyAttribution(maps, trained, 'gradcam').map), [maps]); // eslint-disable-line
  const b = useMemo(() => unitScale(toyAttribution(maps, randomW, 'gradcam').map), [maps, randomW]);
  const base = useMemo(() => resize(src.gray, 224, Math.round((224 * src.gray.h) / src.gray.w)), [src.gray]);
  const rho = spearman(a.data, b.data);
  return (
    <Figure title="Sanity check: randomise the weights" prov={['illustrative', 'live']}
      caption={<>A test from Adebayo et al. (2018), shown on the toy model. If an explanation barely changes when the model’s learned weights are replaced by random ones, it is describing the <i>input</i> (edges, bright structures), not the <i>model</i>. Here the feature channels themselves carry image structure, so even random weights give an anatomical-looking map. Spearman ρ between the two maps: <b>{rho.toFixed(2)}</b>. Notebook 06 ran a cascading-randomisation test on the real checkpoints, but its results are not part of this bundle, so only the toy version is shown here.</>}>
      <ImagePicker compact />
      <div className="grid-3">
        <GrayCanvas img={base} overlay={{ heat: a, opacity: 0.6, cmap: 'inferno' }} label="‘trained’ toy weights" />
        <GrayCanvas img={base} overlay={{ heat: b, opacity: 0.6, cmap: 'inferno' }} label={`random weights · seed ${seed}`} />
        <div className="stack">
          <button className="btn" onClick={() => setSeed((s) => s + 1)}>Draw new random weights</button>
          <span className="small">Weights used: [{randomW.map((v) => v.toFixed(2)).join(', ')}]</span>
          <span className="small muted">High ρ across many draws means the map is mostly a function of the image.</span>
        </div>
      </div>
    </Figure>
  );
}
