import { useEffect, useMemo, useState } from 'react';
import { ImagePicker } from '../components/ImagePicker';
import { Empty, Figure, GrayCanvas, MODEL_META, Seg, Slider, fmtM } from '../components/ui';
import { artifactUrl, loadJSON, useArtifacts } from '../data/artifacts';
import { useImageSource } from '../data/imageSource';
import type { ModelId } from '../data/types';
import { VIT_B16, vitParams } from '../lib/architectures';
import { Gray, makeGray, resize } from '../lib/imageOps';
import { cosine, entropy, patchDescriptors, sincos2d, toyAttentionRow } from '../lib/vit';

const G = 14, P = 16;

function heatFromVector(v: ArrayLike<number>, grid = G): Gray {
  const g = makeGray(grid, grid);
  let mx = 0; for (let i = 0; i < v.length; i++) mx = Math.max(mx, v[i]);
  for (let i = 0; i < grid * grid; i++) g.data[i] = mx ? v[i] / mx : 0;
  return g;
}

function highlightPatch(g: Gray, idx: number | null): Gray {
  if (idx == null) return g;
  const o = { w: g.w, h: g.h, data: new Float32Array(g.data) };
  const px = idx % G, py = Math.floor(idx / G);
  for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
    if (x === 0 || y === 0 || x === P - 1 || y === P - 1) o.data[(py * P + y) * g.w + px * P + x] = 1;
  }
  return o;
}

export function PatchExplorer() {
  const src = useImageSource();
  const img = useMemo(() => resize(src.gray, 224, 224, 'nearest'), [src.gray]);
  const [sel, setSel] = useState<number>(7 * G + 7);
  const patch = useMemo(() => {
    const p = makeGray(P, P);
    const px = sel % G, py = Math.floor(sel / G);
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) p.data[y * P + x] = img.data[(py * P + y) * 224 + px * P + x];
    return p;
  }, [img, sel]);
  const vec = useMemo(() => {
    // flattened (p, p, C) vector with grey replicated into 3 channels → 768 values
    const v = makeGray(768, 1);
    for (let i = 0; i < 256; i++) for (let c = 0; c < 3; c++) v.data[i * 3 + c] = patch.data[i];
    return v;
  }, [patch]);
  const pos = useMemo(() => sincos2d(768, G, true), []);
  const sim = useMemo(() => heatFromVector(pos.slice(1).map((r) => (cosine(r, pos[sel + 1]) + 1) / 2)), [pos, sel]);
  const vp = vitParams(VIT_B16, 6);
  return (
    <Figure title="From an image to 197 tokens" prov="live"
      caption={<>Click any patch. The 224×224 input is cut into 14×14 = 196 non-overlapping 16×16 patches. Each patch is flattened to a 16·16·3 = 768-value vector (grey replicated to RGB) and linearly projected by <i>E</i> ∈ ℝ<sup>768×768</sup>, implemented as a 16×16 convolution with stride 16. A learnable [CLS] token is prepended. <b>Right</b>: cosine similarity between the clicked patch’s fixed 2D sin-cos positional embedding and every other position. This is exactly the encoding MAE and USF-MAE use, computed here. The supervised ViT-B/16 instead <i>learns</i> its positional embeddings.</>}>
      <ImagePicker compact />
      <div className="grid-3">
        <GrayCanvas img={highlightPatch(img, sel)} pixelated onClick={(x, y) => setSel(Math.min(G - 1, Math.floor(y / P)) * G + Math.min(G - 1, Math.floor(x / P)))} label="click a patch" />
        <div className="stack">
          <GrayCanvas img={patch} pixelated label={`patch ${sel} (row ${Math.floor(sel / G)}, col ${sel % G})`} style={{ maxWidth: '11rem' }} />
          <div>
            <span className="small muted">flattened x<sub>p</sub> ∈ ℝ<sup>768</sup></span>
            <GrayCanvas img={vec} pixelated style={{ height: 18 }} />
          </div>
          <table className="data small">
            <tbody>
              <tr><td>Patch embedding (conv 16×16/16)</td><td className="num">{fmtM(vp.patchEmbed)}</td></tr>
              <tr><td>[CLS] + positional table (197×768)</td><td className="num">{fmtM(vp.tokens)}</td></tr>
              <tr><td>12 encoder blocks</td><td className="num">{fmtM(vp.blocks)}</td></tr>
              <tr><td>Final norm + 6-class head</td><td className="num">{fmtM(vp.norm + vp.head)}</td></tr>
              <tr><td><b>Total</b></td><td className="num"><b>{fmtM(vp.total)}</b></td></tr>
            </tbody>
          </table>
        </div>
        <GrayCanvas img={sim} cmap="viridis" pixelated label="sin-cos position similarity" />
      </div>
    </Figure>
  );
}

interface ClsAttention { blocks: number; heads: number; grid: number; cls: number[][][]; rollout?: number[]; }

function useFullAttention(path?: string) {
  const [buf, setBuf] = useState<Uint16Array | null>(null);
  useEffect(() => {
    let c = false; setBuf(null);
    // little-endian uint16, raw attention probability × 65535 (written by omari_us.export)
    if (path) fetch(artifactUrl(path)).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject())).then((b) => !c && setBuf(new Uint16Array(b))).catch(() => {});
    return () => { c = true; };
  }, [path]);
  return buf;
}

export function AttentionExplorer() {
  const src = useImageSource();
  const { manifest } = useArtifacts();
  const img = useMemo(() => resize(src.gray, 224, 224, 'bilinear'), [src.gray]);
  const vitModels = (['vit_b16', 'usf_mae'] as ModelId[]).filter((m) => src.sample?.attention?.[m]);
  const [mode, setMode] = useState<'toy' | 'model'>('toy');
  const [model, setModel] = useState<ModelId>('vit_b16');
  const [block, setBlock] = useState(11);
  const [head, setHead] = useState<number | 'mean'>('mean');
  const [query, setQuery] = useState<number | 'cls'>('cls');
  const [posW, setPosW] = useState(1.2);
  const [temp, setTemp] = useState(0.6);
  const entry = src.sample?.attention?.[model];
  const [cls, setCls] = useState<ClsAttention | null>(null);
  useEffect(() => { let c = false; setCls(null); if (entry) loadJSON<ClsAttention>(entry.cls).then((j) => !c && j && setCls(j)); return () => { c = true; }; }, [entry]);
  const fullMeta = entry?.full?.find((f) => f.block === block);
  const full = useFullAttention(fullMeta?.path);
  const usingModel = mode === 'model' && vitModels.includes(model) && cls;

  const desc = useMemo(() => patchDescriptors(img, P), [img]);
  const toyRow = useMemo(() => toyAttentionRow(desc, G, query === 'cls' ? 7 * G + 7 : query, posW, temp), [desc, query, posW, temp]);

  const row: number[] | null = useMemo(() => {
    if (!usingModel) return toyRow;
    const heads = cls!.heads;
    const pick = (h: number): number[] | null => {
      if (query === 'cls') return cls!.cls[block]?.[h] ?? null;
      if (!full || !fullMeta) return null;
      const T = fullMeta.tokens, base = h * T * T + (query + 1) * T;
      return Array.from({ length: T - 1 }, (_, j) => full[base + 1 + j] / 65535);
    };
    if (head === 'mean') {
      const rows = Array.from({ length: heads }, (_, h) => pick(h));
      if (rows.some((r) => !r)) return null;
      return rows[0]!.map((_, j) => rows.reduce((a, r) => a + r![j], 0) / heads);
    }
    return pick(head);
  }, [usingModel, toyRow, cls, block, head, query, full, fullMeta]);

  const heat = row ? heatFromVector(row) : null;
  const H = row ? entropy(row.map((v) => v / (row.reduce((a, b) => a + b, 0) || 1))) : 0;
  const modelInfo = manifest?.models.find((m) => m.id === model);

  return (
    <Figure title="Self-attention: who does each token look at?" prov={usingModel ? 'model' : 'illustrative'} provDetail={usingModel ? modelInfo?.checkpoint_id : undefined}
      caption={usingModel ? <>Attention weights from <b>{MODEL_META[model].name}</b>, block {block + 1}, {head === 'mean' ? 'mean over heads' : `head ${(head as number) + 1}`}, query = {query === 'cls' ? '[CLS]' : `patch ${query}`}. Rows of softmax(QKᵀ/√d) are exported by the pipeline with forward hooks (full patch-to-patch matrices only for selected blocks; patch-query rows are shown over patch keys, excluding the weight on [CLS]). Attention says where information is <i>mixed</i>; it is not, by itself, evidence of what drives the prediction.</> :
        <><b>Toy attention, not a trained model.</b> Queries and keys are four hand-crafted patch statistics (mean, standard deviation, horizontal and vertical gradient energy) plus a sin-cos position code. It shows the mechanism, a softmax over similarities, and how a position term creates locality. Switch to “Model” once attention maps are exported for this image.</>}>
      <ImagePicker compact />
      <div className="split-wide">
        <div className="controls">
          <Seg label="Source" value={mode} onChange={setMode} options={[{ value: 'toy', label: 'Toy (illustrative)' }, { value: 'model', label: 'Model (exported)', disabled: !vitModels.length }]} />
          {mode === 'model' && vitModels.length > 0 && <>
            <Seg label="Model" value={model} onChange={setModel} options={vitModels.map((m) => ({ value: m, label: MODEL_META[m].name }))} />
            <Slider label="Block" value={block} min={0} max={(cls?.blocks ?? 12) - 1} onChange={setBlock} format={(v) => `${v + 1} of ${cls?.blocks ?? 12}`} />
            <Seg label="Head" value={head} onChange={setHead} options={[{ value: 'mean' as const, label: 'mean' }, ...Array.from({ length: cls?.heads ?? 12 }, (_, h) => ({ value: h, label: String(h + 1) }))]} />
          </>}
          {mode === 'toy' && <>
            <Slider label="Position term weight" value={posW} min={0} max={4} step={0.1} onChange={setPosW} format={(v) => v.toFixed(1)} />
            <Slider label="Temperature T" value={temp} min={0.1} max={3} step={0.05} onChange={setTemp} format={(v) => v.toFixed(2)} />
          </>}
          <div className="row">
            <button className="btn" onClick={() => setQuery('cls')} aria-pressed={query === 'cls'}>{usingModel ? 'Query = [CLS]' : 'Query = centre patch'}</button>
            <span className="small muted">or click a patch</span>
          </div>
          <span className="small">Entropy {H.toFixed(2)} nats (uniform = {Math.log(196).toFixed(2)})</span>
          {mode === 'model' && query !== 'cls' && !fullMeta && <span className="small" style={{ color: 'var(--p-missing)' }}>Patch-to-patch attention was not exported for block {block + 1}; available: {entry?.full?.map((f) => f.block + 1).join(', ') || 'none'}.</span>}
        </div>
        <div className="grid-2">
          <GrayCanvas img={highlightPatch(img, query === 'cls' ? null : query)} onClick={(x, y) => setQuery(Math.min(G - 1, Math.floor(y / P)) * G + Math.min(G - 1, Math.floor(x / P)))} label="query (click)" />
          {heat ? <GrayCanvas img={img} overlay={{ heat, opacity: 0.7, cmap: 'inferno' }} label="attention weights" /> : <Empty title="Not available">This combination was not exported.</Empty>}
        </div>
      </div>
      {usingModel && cls && (
        <div style={{ marginTop: '1rem' }}>
          <span className="small muted">[CLS] attention across depth ({head === 'mean' ? 'head mean' : `head ${(head as number) + 1}`}). Click to select a block.</span>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cls.blocks}, 1fr)`, gap: 3, marginTop: 4 }}>
            {cls.cls.map((b, bi) => {
              const v = head === 'mean' ? b[0].map((_, j) => b.reduce((a, r) => a + r[j], 0) / b.length) : b[head as number];
              return <div key={bi} onClick={() => setBlock(bi)} style={{ outline: bi === block ? '2px solid var(--focus)' : undefined, cursor: 'pointer' }}><GrayCanvas img={heatFromVector(v)} cmap="inferno" pixelated label={String(bi + 1)} /></div>;
            })}
          </div>
        </div>
      )}
    </Figure>
  );
}
