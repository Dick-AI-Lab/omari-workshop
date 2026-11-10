import { useEffect, useMemo, useState } from 'react';
import { ImagePicker } from '../components/ImagePicker';
import { Empty, Figure, GrayCanvas, MODEL_META, Slider, pct } from '../components/ui';
import { loadGrayPNG } from '../data/artifacts';
import { useImageSource } from '../data/imageSource';
import { CLASS_SHORT } from '../data/literature';
import type { ModelId } from '../data/types';
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
  useEffect(() => { let c = false; setG(null); if (path) loadGrayPNG(path).then((x) => !c && setG(x)).catch(() => {}); return () => { c = true; }; }, [path]);
  return g;
}

function MapSimilarity({ a, b }: { a?: string; b?: string }) {
  const ga = useGray(a), gb = useGray(b);
  if (!ga || !gb) return <span className="muted">—</span>;
  const rb = gb.w === ga.w && gb.h === ga.h ? gb : resize(gb, ga.w, ga.h);
  return <span>{spearman(ga.data, rb.data).toFixed(2)}</span>;
}

export function DetectiveLab() {
  const src = useImageSource();
  const [chosen, setChosen] = useState<string[]>([]);
  const [locked, setLocked] = useState(false);
  const [thresh, setThresh] = useState(0.92);
  const s = src.sample;
  const perts = s?.perturbations ?? [];
  const mask = useMemo(() => annotationMask(src.gray, thresh, 1), [src.gray, thresh]);
  const maskImg = useMemo(() => { const g = makeGray(src.gray.w, src.gray.h); mask.forEach((m, i) => (g.data[i] = m)); return g; }, [mask, src.gray]);
  const frac = mask.reduce((a: number, b) => a + b, 0) / mask.length;
  const models = (Object.keys(MODEL_META) as ModelId[]).filter((m) => s?.predictions[m]);
  const baseAttr = (m: ModelId) => s?.attributions?.find((a) => a.model === m && (a.method === 'gradcam' || a.method === 'vit_gradcam') && a.target_class === s.label)?.png;

  return (
    <Figure title="Model detective" prov={perts.length ? ['live', 'model'] : ['live', 'missing']}
      caption={<>Commit to hypotheses <i>before</i> looking at the evidence; the reveal is locked until you do. The annotation detector (bright, saturated pixels) runs live on the selected image. Perturbation evidence comes only from controlled experiments run by the export pipeline on the real checkpoints: each row changes one factor and records every model’s prediction. Map stability is the Spearman rank correlation between the original and perturbed Grad-CAM maps, computed in your browser from the exported PNGs.</>}>
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
            {locked && <button className="btn" onClick={() => { setLocked(false); setChosen([]); }}>Start again</button>}
          </div>
          <b className="small" style={{ marginTop: '0.6rem' }}>2 · Where are the overlays? <span className="muted">(live)</span></b>
          <div className="grid-2">
            <GrayCanvas img={src.gray} overlay={{ heat: maskImg, opacity: 0.85, cmap: 'inferno', threshold: 0.5 }} label={`${pct(frac, 2)} of pixels flagged`} />
            <Slider label="Brightness threshold" value={thresh} min={0.7} max={0.99} step={0.01} onChange={setThresh} format={(v) => v.toFixed(2)} />
          </div>
        </div>
        <div className="stack">
          <b className="small">3 · Evidence</b>
          {!locked ? <Empty title="Locked">Choose at least one hypothesis first.</Empty> :
            !perts.length ? (
              <Empty title={s ? 'No controlled perturbation experiments exported for this sample' : 'Select an exported sample'}>
                The pipeline runs, for each detective sample: contrast ×0.6/×1.5, brightness ±0.15, speckle 0.3/0.6, resolution ½ and ¼, annotation inpainting, and occlusion of the top-attributed region. Enable <code>perturbations: true</code> in <code>configs/export.yaml</code>.
              </Empty>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="data small">
                  <thead><tr><th>Change</th>{models.map((m) => <th key={m} style={{ color: MODEL_META[m].text }}>{MODEL_META[m].name}<br /><span className="muted">p(true) · pred</span></th>)}<th>Map ρ</th></tr></thead>
                  <tbody>
                    <tr><td>none (original)</td>{models.map((m) => <td key={m}>{pct(s!.predictions[m]!.probs[s!.label])} · {CLASS_SHORT[s!.predictions[m]!.pred]}</td>)}<td>1.00</td></tr>
                    {perts.map((p) => (
                      <tr key={p.type + p.param}>
                        <td>{p.label}</td>
                        {models.map((m) => {
                          const pr = p.predictions[m]; const b = s!.predictions[m]!;
                          if (!pr) return <td key={m}>—</td>;
                          const flip = pr.pred !== b.pred;
                          const d = pr.probs[s!.label] - b.probs[s!.label];
                          return <td key={m} style={{ color: flip ? 'var(--p-missing)' : undefined, fontWeight: flip ? 600 : 400 }}>{pct(pr.probs[s!.label])} ({d >= 0 ? '+' : ''}{(d * 100).toFixed(1)}) · {CLASS_SHORT[pr.pred]}</td>;
                        })}
                        <td>{models[0] && p.attributions?.[models[0]] ? <MapSimilarity a={baseAttr(models[0])} b={p.attributions[models[0]]!.png} /> : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="small muted">Red = the predicted class changed. Map ρ uses {models[0] ? MODEL_META[models[0]].name : '—'}’s Grad-CAM for the true class. Compare the results with the hypotheses you locked in.</p>
              </div>
            )}
        </div>
      </div>
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
      caption={<>A test from Adebayo et al. (2018), shown on the toy model. If an explanation barely changes when the model’s learned weights are replaced by random ones, it is describing the <i>input</i> (edges, bright structures), not the <i>model</i>. Here the feature channels themselves carry image structure, so even random weights give an anatomical-looking map. Spearman ρ between the two maps: <b>{rho.toFixed(2)}</b>. Notebook 06 runs the same test on the real checkpoints by cascading randomisation from the head downwards.</>}>
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
