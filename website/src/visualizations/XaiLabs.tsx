import { useEffect, useMemo, useState } from 'react';
import { LineChart } from '../components/charts';
import { ImagePicker } from '../components/ImagePicker';
import { Empty, Figure, GrayCanvas, MODEL_META, Seg, Slider } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { useImageSource } from '../data/imageSource';
import { CLASS_SHORT } from '../data/literature';
import type { AttributionEntry, ModelId, SampleInfo } from '../data/types';
import { layerDepth, layerLabel, layerShort, methodLong, methodShort } from '../data/labels';
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
  grad_rollout: { name: 'Gradient-weighted attention rollout', models: ['vit_b16', 'usf_mae'], kind: 'class-gradient', status: 'implemented', note: 'Rollout of head-averaged ReLU(∇A ⊙ A) across blocks. A simplified variant inspired by Chefer et al. (2021) without layer-wise relevance propagation (LRP); exported as “chefer” in schema 1.0 bundles.' },
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
  const [err, setErr] = useState(false);
  useEffect(() => {
    let c = false; setG(null); setErr(false);
    if (path) loadGrayPNG(path).then((x) => !c && setG(x)).catch((e) => { console.error('[xai] heatmap', path, e); if (!c) setErr(true); });
    return () => { c = true; };
  }, [path]);
  return { g, err };
}

const fmtRaw = (v: number) => (Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(2) : v.toFixed(3));

function HeatTile({ base, entry, opacity, cmap, threshold, label, note }: { base: Gray; entry?: AttributionEntry; opacity: number; cmap: ColormapName; threshold: number; label: string; note?: string }) {
  const { g: heat, err } = useHeat(entry?.png);
  if (!entry) return <div><Empty title="not exported">{label}</Empty></div>;
  if (err) return <div><Empty title="heatmap file could not be loaded" error>{entry.png}</Empty></div>;
  return (
    <div>
      <GrayCanvas img={base} overlay={heat ? { heat, opacity, cmap, threshold } : undefined} label={label} />
      <div className="small muted" style={{ fontFamily: 'var(--sans)', marginTop: '0.2rem' }}>
        {entry.layer_label ?? layerLabel(entry.model, entry.layer)} · {entry.resolution.join('×')} · raw range [{fmtRaw(entry.raw_min)}, {fmtRaw(entry.raw_max)}]{note ? ` · ${note}` : ''}
      </div>
    </div>
  );
}

/** cross-architecture method equivalents (CNN ↔ token version); attention methods only exist for ViTs */
function equivalentMethod(method: string, to: ModelId): string {
  const vit = to !== 'densenet161';
  if (vit) return method === 'cam' ? 'vit_gradcam' : method.startsWith('vit_') || ['attention_rollout', 'grad_rollout'].includes(method) ? method : `vit_${method}`;
  return method.startsWith('vit_') ? method.replace('vit_', '') : method;
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
  const tgt = target === 'pred' ? (pred != null && targets.includes(pred) ? pred : targets[0]) : target;
  const layers = Array.from(new Set(atts.filter((a) => a.model === mdl && a.method === meth && a.target_class === tgt).map((a) => a.layer))).sort((a, b) => layerDepth(a) - layerDepth(b));
  const [layer, setLayer] = useState('');
  const lay = layers.includes(layer) ? layer : layers[layers.length - 1];
  const [opacity, setOpacity] = useState(0.55);
  const [cmap, setCmap] = useState<ColormapName>('inferno');
  const [threshold, setThreshold] = useState(0);
  const [chipFilter, setChipFilter] = useState<'all' | 'correct' | 'wrong'>('all');
  const base = useMemo(() => resize(src.gray, 224, 224, 'bilinear'), [src.gray]);
  /** exact match on layer when given; otherwise the deepest available layer. Returns whether the layer matched. */
  const find = (m: ModelId, me: string, t: number, l?: string): { e?: AttributionEntry; exact: boolean } => {
    const cands = atts.filter((a) => a.model === m && a.method === me && a.target_class === t);
    const exact = l ? cands.find((a) => a.layer === l) : undefined;
    if (exact) return { e: exact, exact: true };
    const deepest = [...cands].sort((a, b) => layerDepth(b.layer) - layerDepth(a.layer))[0];
    return { e: deepest, exact: !l };
  };
  const allChips = manifest?.samples.filter((s) => s.attributions?.length) ?? [];
  const chipOk = (s: SampleInfo) => { const p = s.predictions[mdl ?? 'densenet161']; return p ? p.pred === s.label : undefined; };
  const sampleChips = allChips.filter((s) => chipFilter === 'all' || (chipFilter === 'correct' ? chipOk(s) === true : chipOk(s) === false));

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
  const entry = mdl && meth && tgt != null ? find(mdl, meth, tgt, lay).e : undefined;
  const ok = pred != null ? pred === src.sample!.label : undefined;
  return (
    <Figure title="Explainability laboratory" prov="model" provDetail={entry ? `${entry.provenance.script} · ${entry.provenance.checkpoint_id ?? ''} · layer ${entry.layer}${entry.legacy_method ? ` · exported as '${entry.legacy_method}'` : ''}` : undefined}
      caption={<>Heatmaps from the fine-tuned checkpoints for <b>{src.sample?.id}</b> (expert label {CLASS_SHORT[src.sample!.label]}{pred != null ? <>; {MODEL_META[mdl].name} predicts {CLASS_SHORT[pred]}, {ok ? 'correct' : <b>incorrect</b>}</> : ''}). Each map is min–max scaled for display; the raw range under each tile shows how strong the evidence actually was, because scaling makes a weak, diffuse map look as confident as a strong one. The threshold hides values below a fraction of the maximum. Class-gradient methods answer “what increases <i>this</i> class score?”; attention rollout is not class-specific. ViT layer labels count blocks from 1: “after block 8 of 12” is the token representation entering block 9. Use the chips to compare correct and incorrect predictions.</>}>
      <ImagePicker compact />
      <div className="row" style={{ gap: '0.3rem', marginBottom: '0.6rem' }}>
        <Seg label={`Samples (✓/✗ = ${MODEL_META[mdl].name} correct/wrong)`} value={chipFilter} onChange={setChipFilter} options={[
          { value: 'all', label: `All (${allChips.length})` },
          { value: 'correct', label: `Correct (${allChips.filter((s) => chipOk(s) === true).length})` },
          { value: 'wrong', label: `Incorrect (${allChips.filter((s) => chipOk(s) === false).length})` },
        ]} />
      </div>
      <div className="row" style={{ gap: '0.3rem', marginBottom: '0.6rem' }}>
        {sampleChips.map((s) => {
          const o = chipOk(s);
          return <button key={s.id} className="btn" title={s.id} style={{ padding: '0.1rem 0.45rem', borderColor: o === false ? 'var(--p-missing)' : undefined }} aria-pressed={src.sample?.id === s.id} onClick={() => src.selectSample(s.id)}>
            {CLASS_SHORT[s.label]}{o === false ? ' ✗' : o ? ' ✓' : ''}</button>;
        })}
      </div>
      <div className="split-wide">
        <div className="controls">
          <Seg label="Compare" value={compare} onChange={setCompare} options={[{ value: 'methods', label: 'Methods for one model' }, { value: 'models', label: 'Architectures for one method' }]} />
          <Seg label="Model" value={mdl} onChange={setModel} options={models.map((m) => ({ value: m, label: MODEL_META[m].name }))} />
          <Seg label="Method" value={meth} onChange={setMethod} options={methods.map((m) => ({ value: m, label: methodShort(m), title: methodLong(m) }))} />
          <Seg label="Target class" value={target} onChange={setTarget} options={[{ value: 'pred' as const, label: `Predicted (${pred != null ? CLASS_SHORT[pred] : '—'})` }, ...targets.map((t) => ({ value: t, label: CLASS_SHORT[t] }))]} />
          {layers.length > 0 && <Seg label="Layer (shallow → deep)" value={lay} onChange={setLayer} options={layers.map((l) => ({ value: l, label: layerShort(l, atts.find((a) => a.layer === l)?.layer_label), title: layerLabel(mdl, l) }))} />}
          <Slider label="Opacity" value={opacity} min={0} max={1} step={0.05} onChange={setOpacity} format={(v) => v.toFixed(2)} />
          <Slider label="Threshold" value={threshold} min={0} max={0.9} step={0.05} onChange={setThreshold} format={(v) => `${Math.round(v * 100)}% of max`} />
          <Seg label="Colour scale" value={cmap} onChange={setCmap} options={(['inferno', 'viridis', 'gray', 'jet'] as ColormapName[]).map((c) => ({ value: c, label: c }))} />
          <div style={{ height: 8, background: cssGradient(cmap), borderRadius: 2 }} aria-hidden />
          {cmap === 'jet' && <span className="small" style={{ color: 'var(--p-missing)' }}>Jet’s luminance is not monotone: mid-range values look brightest and create false boundaries.</span>}
          {meth && <span className="small muted"><b>{methodLong(meth)}.</b> {METHOD_INFO[meth]?.note}</span>}
        </div>
        <div className="grid-2">
          <HeatTile base={base} entry={entry} opacity={opacity} cmap={cmap} threshold={threshold} label={`${MODEL_META[mdl].name} · ${methodShort(meth)} · ${CLASS_SHORT[tgt]}`} />
          {compare === 'methods'
            ? methods.filter((m) => m !== meth).slice(0, 3).map((m) => {
              const r = find(mdl, m, tgt, lay);
              return <HeatTile key={m} base={base} entry={r.e} opacity={opacity} cmap={cmap} threshold={threshold} label={methodShort(m)} note={r.e && !r.exact ? 'selected layer not exported for this method' : undefined} />;
            })
            : models.filter((m) => m !== mdl).map((m) => {
              const alt = equivalentMethod(meth, m);
              // same layer id where the architecture has it (ViT ↔ ViT); otherwise the deepest exported layer
              const r = find(m, alt, tgt, lay);
              return <HeatTile key={m} base={base} entry={r.e} opacity={opacity} cmap={cmap} threshold={threshold} label={`${MODEL_META[m].name} · ${methodShort(alt)}`} note={r.e && !r.exact ? 'deepest exported layer (no equivalent of the selected layer)' : undefined} />;
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
            <tr key={k}><td title={methodLong(k)}>{v.name}</td><td className="small">{v.kind}</td>{ids.map((m) => <td key={m}>{v.models.includes(m) ? (v.status === 'reserved' ? 'reserved' : '✓') : '—'}</td>)}</tr>
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

@register_method("metacampp",
                 models=("densenet161", "vit_b16", "usf_mae"))
def metacampp(model, x, target_class, layer=None,
              **kw) -> torch.Tensor:
    """Return a non-negative (H', W') map for one image."""
    raise NotImplementedError  # supplied by the MetaCAM++ authors`}</pre>
      </Empty>
    </Figure>
  );
}
