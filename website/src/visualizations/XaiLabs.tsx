import { useEffect, useMemo, useState } from 'react';
import { LineChart } from '../components/charts';
import { ImagePicker } from '../components/ImagePicker';
import { Empty, Figure, GrayCanvas, MODEL_META, Seg, Slider } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { useImageSource } from '../data/imageSource';
import { CLASS_SHORT } from '../data/literature';
import type { AttributionEntry, ModelId } from '../data/types';
import { ColormapName, cssGradient } from '../lib/colormaps';
import { Gray, resize } from '../lib/imageOps';
import { TOY_CHANNELS, ToyMethod, toyAttribution, toyDeletionCurve, toyFeatureMaps, toyScore, unitScale } from '../lib/toyModels';

export const METHOD_INFO: Record<string, { name: string; models: ModelId[]; kind: 'class-gradient' | 'class-weights' | 'attention' | 'reserved'; status: 'implemented' | 'reserved'; note: string }> = {
  cam: { name: 'CAM', models: ['densenet161'], kind: 'class-weights', status: 'implemented', note: 'Needs conv features → GAP → linear. Exact for DenseNet-161’s head.' },
  gradcam: { name: 'Grad-CAM', models: ['densenet161'], kind: 'class-gradient', status: 'implemented', note: 'Any conv layer; gradients of the class logit, spatially averaged.' },
  gradcampp: { name: 'Grad-CAM++', models: ['densenet161'], kind: 'class-gradient', status: 'implemented', note: 'Pixel-wise weighting of positive gradients (exp-score closed form).' },
  vit_gradcam: { name: 'Grad-CAM (tokens)', models: ['vit_b16', 'usf_mae'], kind: 'class-gradient', status: 'implemented', note: 'Applied to patch tokens of a late block, reshaped 196 → 14×14; [CLS] dropped.' },
  vit_gradcampp: { name: 'Grad-CAM++ (tokens)', models: ['vit_b16', 'usf_mae'], kind: 'class-gradient', status: 'implemented', note: 'Same token reshape as above.' },
  attention_rollout: { name: 'Attention rollout', models: ['vit_b16', 'usf_mae'], kind: 'attention', status: 'implemented', note: 'Products of (A + I)/2 across blocks. Not class-specific.' },
  chefer: { name: 'Gradient-weighted rollout', models: ['vit_b16', 'usf_mae'], kind: 'class-gradient', status: 'implemented', note: 'Rollout of ReLU(∇A ⊙ A) averaged over heads (Chefer et al. 2021, simplified, no LRP).' },
  metacampp: { name: 'MetaCAM++', models: ['densenet161', 'vit_b16', 'usf_mae'], kind: 'reserved', status: 'reserved', note: 'Research method in development. Integration interface only; no implementation or results included.' },
};

/* ------------------------------------------------------------------ */
export function CamToy({ shortcut = false }: { shortcut?: boolean }) {
  const src = useImageSource();
  const maps = useMemo(() => toyFeatureMaps(src.gray), [src.gray]);
  const presetA = [1.2, -0.6, 0.8, 0.3, 0.2, 0];
  const presetB = [0.1, 0, 0.1, 0.1, 0, 2.5];
  const [w, setW] = useState<number[]>(shortcut ? presetB : presetA);
  const [method, setMethod] = useState<ToyMethod>('gradcam');
  const [opacity, setOpacity] = useState(0.6);
  const res = useMemo(() => toyAttribution(maps, w, method), [maps, w, method]);
  const shown = useMemo(() => unitScale(res.map, method === 'cam'), [res, method]);
  const base = useMemo(() => resize(src.gray, 224, Math.round(224 * src.gray.h / src.gray.w), 'bilinear'), [src.gray]);
  const del = useMemo(() => toyDeletionCurve(maps, w, res.map), [maps, w, res]);
  const rnd = useMemo(() => {
    // deletion with a random order, as a reference
    const r = { w: res.map.w, h: res.map.h, data: res.map.data.map((_, i) => Math.sin(i * 12.9898) * 43758.5453 % 1) } as Gray;
    return toyDeletionCurve(maps, w, r);
  }, [maps, w, res]);
  const score = toyScore(maps, w);
  return (
    <Figure title={shortcut ? 'A toy model that has learned a shortcut' : 'CAM, Grad-CAM and Grad-CAM++ on a model you can read'} prov={['illustrative', 'live']}
      caption={<>A <b>toy</b> classifier: six fixed, hand-crafted feature channels <i>A<sub>k</sub></i> (14×14, non-negative) are pooled and combined linearly, <i>y</i> = Σ<sub>k</sub> w<sub>k</sub>·GAP(A<sub>k</sub>). You set the weights, so you know exactly what the model “uses”, and can check whether the explanation recovers it. For this head Grad-CAM equals ReLU(CAM) up to scale; Grad-CAM++ re-weights channels by how concentrated their activation is. The deletion curve removes the highest-ranked cells first: the faster the score falls relative to random order, the more faithful the map is <i>for this model</i>.</>}>
      <ImagePicker compact />
      <div className="split-wide">
        <div className="controls">
          {TOY_CHANNELS.map((c, k) => (
            <div key={c.id} className="row" style={{ gap: '0.5rem', flexWrap: 'nowrap' }}>
              <div style={{ width: 44, flex: 'none' }}><GrayCanvas img={maps[k]} cmap="inferno" pixelated /></div>
              <div style={{ flex: 1 }}><Slider label={`w · ${c.label}`} value={w[k]} min={-2} max={3} step={0.1} onChange={(v) => setW(w.map((x, j) => (j === k ? v : x)))} format={(v) => v.toFixed(1)} /></div>
            </div>
          ))}
          <div className="row">
            <button className="btn" onClick={() => setW(presetA)}>Anatomy-like weights</button>
            <button className="btn" onClick={() => setW(presetB)}>Shortcut on overlays</button>
          </div>
        </div>
        <div className="stack">
          <div className="row">
            <Seg label="Method" value={method} onChange={setMethod} options={[{ value: 'cam', label: 'CAM' }, { value: 'gradcam', label: 'Grad-CAM' }, { value: 'gradcampp', label: 'Grad-CAM++' }]} />
            <div style={{ width: '10rem' }}><Slider label="Opacity" value={opacity} min={0} max={1} step={0.05} onChange={setOpacity} format={(v) => v.toFixed(2)} /></div>
          </div>
          <div className="grid-2">
            <GrayCanvas img={base} overlay={{ heat: shown, opacity, cmap: method === 'cam' ? 'diverging' : 'inferno' }} label={`toy score y = ${score.toFixed(3)}`} />
            <div>
              <LineChart width={320} height={210} xLabel="fraction of cells removed" yLabel="toy score"
                series={[{ name: 'by attribution', color: 'var(--p-illus)', x: del.map((_, i) => i / (del.length - 1)), y: del }, { name: 'random order', color: 'var(--rule-strong)', x: rnd.map((_, i) => i / (rnd.length - 1)), y: rnd, dashed: true }]}
                xTickFormat={(v) => v.toFixed(1)} yTickFormat={(v) => v.toFixed(2)} />
            </div>
          </div>
          <table className="data small">
            <thead><tr><th>Channel</th>{TOY_CHANNELS.map((c) => <th key={c.id} className="num">{c.label}</th>)}</tr></thead>
            <tbody><tr><td>{method === 'cam' ? 'w_k' : method === 'gradcam' ? 'α_k = w_k / Z' : 'w_k^{++}'}</td>{res.channelWeights.map((v, i) => <td key={i} className="num">{v.toExponential(1)}</td>)}</tr></tbody>
          </table>
        </div>
      </div>
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
function useHeat(path?: string) {
  const [g, setG] = useState<Gray | null>(null);
  useEffect(() => { let c = false; setG(null); if (path) loadGrayPNG(path).then((x) => !c && setG(x)).catch(() => {}); return () => { c = true; }; }, [path]);
  return g;
}

function HeatTile({ base, entry, opacity, cmap, threshold, label }: { base: Gray; entry?: AttributionEntry; opacity: number; cmap: ColormapName; threshold: number; label: string }) {
  const heat = useHeat(entry?.png);
  if (!entry) return <div><Empty title="not exported">{label}</Empty></div>;
  return <GrayCanvas img={base} overlay={heat ? { heat, opacity, cmap, threshold } : undefined} label={label} />;
}

export function XaiLab() {
  const src = useImageSource();
  const { manifest } = useArtifacts();
  const atts = src.sample?.attributions ?? [];
  const models = (Object.keys(MODEL_META) as ModelId[]).filter((m) => atts.some((a) => a.model === m));
  const [model, setModel] = useState<ModelId>('densenet161');
  const [compare, setCompare] = useState<'methods' | 'models'>('methods');
  const mdl = models.includes(model) ? model : models[0];
  const methods = Array.from(new Set(atts.filter((a) => a.model === mdl).map((a) => a.method)));
  const [method, setMethod] = useState('gradcam');
  const meth = methods.includes(method) ? method : methods[0];
  const targets = Array.from(new Set(atts.filter((a) => a.model === mdl && a.method === meth).map((a) => a.target_class))).sort();
  const pred = mdl ? src.sample?.predictions[mdl]?.pred : undefined;
  const [target, setTarget] = useState<number | 'pred'>('pred');
  const tgt = target === 'pred' ? (pred ?? targets[0]) : target;
  const layers = Array.from(new Set(atts.filter((a) => a.model === mdl && a.method === meth && a.target_class === tgt).map((a) => a.layer)));
  const [layer, setLayer] = useState('');
  const lay = layers.includes(layer) ? layer : layers[layers.length - 1];
  const [opacity, setOpacity] = useState(0.55);
  const [cmap, setCmap] = useState<ColormapName>('inferno');
  const [threshold, setThreshold] = useState(0);
  const base = useMemo(() => resize(src.gray, 224, 224, 'bilinear'), [src.gray]);
  const find = (m: ModelId, me: string, t: number, l?: string) => atts.find((a) => a.model === m && a.method === me && a.target_class === t && (!l || a.layer === l)) ?? atts.find((a) => a.model === m && a.method === me && a.target_class === t);
  const sampleChips = manifest?.samples.filter((s) => s.attributions?.length) ?? [];

  if (!atts.length) {
    return (
      <Figure title="Explainability laboratory" prov="missing" caption="Heatmaps here are computed by notebook 06 / the export pipeline from the fine-tuned checkpoints, for each compatible model × method × target class × layer, and stored as 8-bit PNGs with their raw value range.">
        <ImagePicker compact />
        <Empty title={src.kind === 'sample' ? 'No attributions exported for this sample' : 'Select an exported FETAL_PLANES_DB sample'}>
          Run <code>python scripts/export_website_artifacts.py --attributions all</code>. Until then use the toy model above, where the ground truth is known by construction.
        </Empty>
      </Figure>
    );
  }
  const entry = mdl && meth && tgt != null ? find(mdl, meth, tgt, lay) : undefined;
  return (
    <Figure title="Explainability laboratory" prov="model" provDetail={entry ? `${entry.provenance.script} · ${entry.provenance.checkpoint_id ?? ''} · layer ${entry.layer}` : undefined}
      caption={<>Heatmaps from the fine-tuned checkpoints for <b>{src.sample?.id}</b> (expert label {CLASS_SHORT[src.sample!.label]}). Each map is min–max scaled; the threshold hides values below a fraction of the maximum. Class-gradient methods answer “what increases <i>this</i> class score?”. Attention rollout is not class-specific. Compare a correct prediction with an incorrect one using the sample chips.</>}>
      <ImagePicker compact />
      <div className="row" style={{ gap: '0.3rem', marginBottom: '0.6rem' }}>
        {sampleChips.slice(0, 16).map((s) => {
          const p = s.predictions[mdl ?? 'densenet161'];
          const ok = p ? p.pred === s.label : undefined;
          return <button key={s.id} className="btn" style={{ padding: '0.1rem 0.45rem', borderColor: ok === false ? 'var(--p-missing)' : undefined }} aria-pressed={src.sample?.id === s.id} onClick={() => src.selectSample(s.id)}>
            {CLASS_SHORT[s.label]}{ok === false ? ' ✗' : ok ? ' ✓' : ''}</button>;
        })}
      </div>
      <div className="split-wide">
        <div className="controls">
          <Seg label="Compare" value={compare} onChange={setCompare} options={[{ value: 'methods', label: 'Methods for one model' }, { value: 'models', label: 'Models for one method' }]} />
          <Seg label="Model" value={mdl} onChange={setModel} options={models.map((m) => ({ value: m, label: MODEL_META[m].name }))} />
          <Seg label="Method" value={meth} onChange={setMethod} options={methods.map((m) => ({ value: m, label: METHOD_INFO[m]?.name ?? m }))} />
          <Seg label="Target class" value={target} onChange={setTarget} options={[{ value: 'pred' as const, label: `Predicted (${pred != null ? CLASS_SHORT[pred] : '—'})` }, ...targets.map((t) => ({ value: t, label: CLASS_SHORT[t] }))]} />
          {layers.length > 1 && <Seg label="Layer" value={lay} onChange={setLayer} options={layers.map((l) => ({ value: l, label: l.replace('features.', '') }))} />}
          <Slider label="Opacity" value={opacity} min={0} max={1} step={0.05} onChange={setOpacity} format={(v) => v.toFixed(2)} />
          <Slider label="Threshold" value={threshold} min={0} max={0.9} step={0.05} onChange={setThreshold} format={(v) => `${Math.round(v * 100)}% of max`} />
          <Seg label="Colour scale" value={cmap} onChange={setCmap} options={(['inferno', 'viridis', 'gray', 'jet'] as ColormapName[]).map((c) => ({ value: c, label: c }))} />
          <div style={{ height: 8, background: cssGradient(cmap), borderRadius: 2 }} aria-hidden />
          {cmap === 'jet' && <span className="small" style={{ color: 'var(--p-missing)' }}>Jet’s luminance is not monotone: mid-range values look brightest and create false boundaries.</span>}
          {meth && <span className="small muted">{METHOD_INFO[meth]?.note}</span>}
        </div>
        <div className="grid-2">
          <HeatTile base={base} entry={entry} opacity={opacity} cmap={cmap} threshold={threshold} label={`${MODEL_META[mdl].name} · ${METHOD_INFO[meth]?.name ?? meth} · ${CLASS_SHORT[tgt]}`} />
          {compare === 'methods'
            ? methods.filter((m) => m !== meth).slice(0, 3).map((m) => <HeatTile key={m} base={base} entry={find(mdl, m, tgt)} opacity={opacity} cmap={cmap} threshold={threshold} label={METHOD_INFO[m]?.name ?? m} />)
            : models.filter((m) => m !== mdl).map((m) => {
              const alt = meth.startsWith('vit_') || ['attention_rollout', 'chefer'].includes(meth) ? (m === 'densenet161' ? meth.replace('vit_', '') : meth) : (m === 'densenet161' ? meth : `vit_${meth}`);
              return <HeatTile key={m} base={base} entry={find(m, alt, tgt)} opacity={opacity} cmap={cmap} threshold={threshold} label={`${MODEL_META[m].name} · ${METHOD_INFO[alt]?.name ?? alt}`} />;
            })}
        </div>
      </div>
    </Figure>
  );
}

export function CompatibilityTable() {
  const ids: ModelId[] = ['densenet161', 'vit_b16', 'usf_mae'];
  return (
    <Figure title="Which method applies to which architecture?" prov="live" wide={false}
      caption="Implementation status in python/omari_us/explainability. ‘Reserved’ means an integration slot with no implementation.">
      <table className="data">
        <thead><tr><th>Method</th><th>Signal</th>{ids.map((m) => <th key={m} style={{ color: MODEL_META[m].text }}>{MODEL_META[m].name}</th>)}</tr></thead>
        <tbody>
          {Object.entries(METHOD_INFO).map(([k, v]) => (
            <tr key={k}><td>{v.name}</td><td className="small">{v.kind}</td>{ids.map((m) => <td key={m}>{v.models.includes(m) ? (v.status === 'reserved' ? 'reserved' : '✓') : '—'}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </Figure>
  );
}

export function MetaCamPanel() {
  const { manifest } = useArtifacts();
  const has = manifest?.samples.some((s) => s.attributions?.some((a) => a.method === 'metacampp'));
  return (
    <Figure title="MetaCAM++ (research in progress)" prov={has ? 'model' : 'missing'} wide={false}
      caption="This slot is reserved for MetaCAM++. Its formulation, implementation and results are intentionally absent until the reference implementation is provided. Any MetaCAM++ maps exported with method id ‘metacampp’ appear in the laboratory above automatically.">
      <Empty title={has ? 'MetaCAM++ attributions found in the bundle' : 'Awaiting reference implementation'}>
        Register an implementation in Python and re-export:
        <pre className="code" style={{ marginTop: '0.5rem' }}>{`from omari_us.explainability import register_method

@register_method("metacampp", models=("densenet161", "vit_b16", "usf_mae"))
def metacampp(model, x, target_class, layer=None, **kw) -> torch.Tensor:
    """Return a non-negative (H', W') map for one image."""
    raise NotImplementedError  # supplied by the MetaCAM++ authors`}</pre>
      </Empty>
    </Figure>
  );
}
