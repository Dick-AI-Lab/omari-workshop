import { useEffect, useMemo, useState } from 'react';
import { ImagePicker } from '../components/ImagePicker';
import { Empty, Figure, GrayCanvas, Prov, Seg, Slider, fmtM, pct } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { shortSha, usfEpochs } from '../data/labels';
import { usfCheckpointText } from '../components/Disclosure';
import { useImageSource } from '../data/imageSource';
import { maeDecoderParams, vitEncoderMACs, vitParams } from '../lib/architectures';
import { Gray, makeGray, resize } from '../lib/imageOps';
import { randomMask } from '../lib/vit';

const G = 14, P = 16;

function applyMask(img: Gray, mask: boolean[], fill: 'grey' | 'black' = 'grey'): Gray {
  const o = { w: img.w, h: img.h, data: new Float32Array(img.data) };
  mask.forEach((m, i) => {
    if (!m) return;
    const px = i % G, py = Math.floor(i / G);
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) o.data[(py * P + y) * 224 + px * P + x] = fill === 'grey' ? 0.18 + (((x + y) >> 2) % 2) * 0.04 : 0;
  });
  return o;
}

/** NOT USF-MAE: fill each masked patch with the mean of visible 4-neighbour patches (a non-learned baseline). */
function naiveFill(img: Gray, mask: boolean[]): Gray {
  const o = { w: img.w, h: img.h, data: new Float32Array(img.data) };
  const means = mask.map((_, i) => {
    const px = i % G, py = Math.floor(i / G);
    let s = 0; for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) s += img.data[(py * P + y) * 224 + px * P + x];
    return s / (P * P);
  });
  mask.forEach((m, i) => {
    if (!m) return;
    const px = i % G, py = Math.floor(i / G);
    const nb = [[0, -1], [0, 1], [-1, 0], [1, 0]].map(([dx, dy]) => [px + dx, py + dy]).filter(([x, y]) => x >= 0 && y >= 0 && x < G && y < G && !mask[y * G + x]);
    const v = nb.length ? nb.reduce((a, [x, y]) => a + means[y * G + x], 0) / nb.length : 0.2;
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) o.data[(py * P + y) * 224 + px * P + x] = v;
  });
  return o;
}

export function MaskingExplorer() {
  const src = useImageSource();
  const { manifest } = useArtifacts();
  const img = useMemo(() => resize(src.gray, 224, 224, 'nearest'), [src.gray]);
  const primary = src.sample?.reconstructions ?? [];
  const legacy = src.sample?.reconstructions_legacy ?? [];
  // 'primary' = what the bundle marks as the main reconstructions (500ep when generated); 'legacy' = historical 100ep
  const [which, setWhich] = useState<'primary' | 'legacy'>('primary');
  const recs = which === 'legacy' && legacy.length ? legacy : primary;
  const ratios = Array.from(new Set(recs.map((r) => r.mask_ratio))).sort((a, b) => a - b);
  const real = ratios.length > 0;
  const [freeRatio, setFreeRatio] = useState(0.75);
  const [seed, setSeed] = useState(0);
  const [view, setView] = useState<'model' | 'naive'>('model');
  // with exported reconstructions the slider snaps to the exported ratios (index slider); otherwise it is free
  const ratio = real ? ratios.reduce((best, r) => (Math.abs(r - freeRatio) < Math.abs(best - freeRatio) ? r : best), ratios[0]) : freeRatio;
  const exported = real ? recs.find((r) => Math.abs(r.mask_ratio - ratio) < 1e-6) : undefined;
  const mask = useMemo(() => {
    if (exported) { const m = new Array(G * G).fill(false); exported.masked.forEach((i) => (m[i] = true)); return m; }
    return randomMask(G * G, ratio, seed + 1);
  }, [exported, ratio, seed]);
  const [recImg, setRecImg] = useState<Gray | null>(null);
  const [recErr, setRecErr] = useState<string | null>(null);
  useEffect(() => {
    let c = false; setRecImg(null); setRecErr(null);
    if (exported) {
      const p = exported.pasted_png ?? exported.reconstruction_png;
      loadGrayPNG(p).then((g) => !c && setRecImg(g)).catch((e) => { console.error('[mae] reconstruction', p, e); if (!c) setRecErr(p); });
    }
    return () => { c = true; };
  }, [exported]);
  const masked = useMemo(() => applyMask(img, mask), [img, mask]);
  const naive = useMemo(() => naiveFill(img, mask), [img, mask]);
  const visible = mask.filter((m) => !m).length;
  const macsFull = vitEncoderMACs(197), macsVis = vitEncoderMACs(visible + 1);
  const enc = vitParams(undefined, 0), dec = maeDecoderParams();
  const visTokens = useMemo(() => {
    const g = makeGray(G, G);
    mask.forEach((m, i) => (g.data[i] = m ? 0 : 1));
    return g;
  }, [mask]);
  const ckpt = exported?.provenance.checkpoint_id ?? undefined;
  const ep = usfEpochs(ckpt);
  const recProv = exported ? `${ckpt ?? 'checkpoint not recorded'}${exported.provenance.checkpoint_sha256 ? ` · sha256 ${shortSha(exported.provenance.checkpoint_sha256)}…` : ''}` : '';
  const historical = ep === 100;
  const classifierCkpt = usfCheckpointText(manifest);
  return (
    <Figure title="Masked autoencoding: learn by filling in the gaps" prov={exported ? ['live', 'model'] : ['live', 'missing']} provDetail={exported ? `${exported.provenance.script} · ${recProv}` : undefined}
      caption={<>MAE removes a random subset of patches (uniform noise, keep the lowest-scoring 1−r), encodes only the visible patches, and asks a light decoder to predict the pixels of the masked ones. The masking and the compute arithmetic are live. A reconstruction is shown only if it was produced by the USF-MAE decoder for this image and mask{real ? <>; the slider snaps to the exported ratios ({ratios.map((r) => pct(r, 0)).join(', ')}) and uses the exported mask (seed {exported?.seed ?? 0})</> : ''}. The fine-tuned USF-MAE classifier in this workshop was initialised from {classifierCkpt}.{historical && <> <b>The reconstructions shown come from the historical 100-epoch checkpoint</b>, not from the checkpoint the classifier was fine-tuned from.</>} The naive baseline exists only to show what a <i>non-learned</i> fill looks like.</>}>
      <ImagePicker compact />
      <div className="split-wide">
        <div className="controls">
          {legacy.length > 0 && (
            <Seg label="Reconstruction checkpoint" value={which} onChange={setWhich} options={[
              { value: 'primary', label: usfEpochs(primary[0]?.provenance.checkpoint_id) === 500 ? '500-epoch (used for this workshop)' : 'Primary' },
              { value: 'legacy', label: 'Historical, 100-epoch checkpoint' },
            ]} />
          )}
          {real
            ? <Slider label="Mask ratio r (exported values)" value={ratios.indexOf(ratio)} min={0} max={ratios.length - 1} step={1} onChange={(i) => setFreeRatio(ratios[i])} format={(i) => pct(ratios[i] ?? ratio, 0)} />
            : <>
              <Slider label="Mask ratio r" value={freeRatio} min={0} max={0.95} step={0.05} onChange={setFreeRatio} format={(v) => pct(v, 0)} />
              <div className="row"><button className="btn" onClick={() => setSeed((s) => s + 1)}>New random mask</button><span className="small muted">live mask, seed {seed}</span></div>
            </>}
          <Seg label="Masked-patch view" value={view} onChange={setView} options={[{ value: 'model', label: 'USF-MAE reconstruction' }, { value: 'naive', label: 'Naive fill (not MAE)' }]} />
          <table className="data small">
            <tbody>
              <tr><td>Visible tokens to encoder</td><td className="num">{visible} + [CLS]</td></tr>
              <tr><td>Encoder MACs vs full image</td><td className="num">{pct(macsVis / macsFull, 0)}</td></tr>
              <tr><td>Encoder (ViT-B/16) parameters</td><td className="num">{fmtM(enc.total)}</td></tr>
              <tr><td>Decoder (8 blocks, 512-d) parameters</td><td className="num">{fmtM(dec.total)}</td></tr>
            </tbody>
          </table>
          <span className="small muted">USF-MAE uses MAE’s default r = 75% during pretraining, with the loss on raw (not per-patch-normalised) pixels.</span>
        </div>
        <div className="grid-3">
          <GrayCanvas img={img} label="input" />
          <div className="stack" style={{ gap: '0.3rem' }}>
            <GrayCanvas img={masked} label={`${pct(ratio, 0)} masked`} />
            <GrayCanvas img={visTokens} pixelated label="encoder sees" style={{ width: '45%' }} />
          </div>
          <div className="stack" style={{ gap: '0.3rem' }}>
            {view === 'naive' ? <><GrayCanvas img={naive} label="naive neighbour fill" /><Prov kind="illustrative" /></> :
              recImg ? <>
                <GrayCanvas img={recImg} label={`USF-MAE${ep ? ` ${ep}ep` : ''} · MSE ${exported!.loss_masked_mse.toFixed(4)}`} />
                <Prov kind="model" detail={recProv} />
                <span className="small muted">Checkpoint: {recProv}{historical ? ' — historical, 100-epoch checkpoint' : ''}</span>
              </> :
                recErr ? <Empty title="Reconstruction file could not be loaded" error>{recErr}</Empty> :
                <Empty title="No authentic reconstruction for this setting">
                  {src.kind !== 'sample' ? 'Select an exported sample. ' : ''}Reconstructions require the USF-MAE pretraining checkpoint <i>with</i> decoder weights (<code>finalize_website_bundle.py --reconstruct</code>). Nothing is synthesised in its place.
                </Empty>}
          </div>
        </div>
      </div>
    </Figure>
  );
}

export function PretrainFinetuneFigure() {
  const [stage, setStage] = useState<'pretrain' | 'finetune'>('pretrain');
  const W = 720, H = 170;
  const box = (x: number, w: number, label: string, sub: string, on: boolean, color = 'var(--ink)') => (
    <g transform={`translate(${x}, 50)`}>
      <rect width={w} height={62} rx={5} fill={on ? '#fff' : '#f0f3f5'} stroke={on ? color : 'var(--rule-strong)'} strokeWidth={on ? 2 : 1} strokeDasharray={on ? undefined : '4 3'} />
      <text x={w / 2} y={27} textAnchor="middle" style={{ fill: on ? 'var(--ink)' : 'var(--ink-3)', fontWeight: 600, fontSize: 13 }}>{label}</text>
      <text x={w / 2} y={46} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{sub}</text>
    </g>
  );
  const arrow = (x1: number, x2: number) => <path d={`M${x1},81 L${x2 - 6},81`} stroke="var(--ink-2)" markerEnd="url(#ah)" />;
  const pre = stage === 'pretrain';
  return (
    <Figure title="Two stages, one encoder" prov="illustrative"
      caption={<>Schematic. <b>Pretraining</b> (done once by the USF-MAE authors on ≈370k unlabelled ultrasound images; this workshop uses their 500-epoch checkpoint): masked patches in, pixels out, loss only on masked patches. <b>Fine-tuning</b> (what notebook 04 does): discard the decoder, feed all 196 patches, attach a 6-way linear head, and train on labelled FETAL_PLANES_DB images. Frozen-encoder “linear probing” updates only the head; full fine-tuning updates everything, usually with layer-wise learning-rate decay.</>}>
      <Seg label="Stage" value={stage} onChange={setStage} options={[{ value: 'pretrain', label: 'Self-supervised pretraining' }, { value: 'finetune', label: 'Supervised fine-tuning' }]} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="MAE pretraining versus fine-tuning schematic">
        <defs><marker id="ah" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--ink-2)" /></marker></defs>
        {/* boxes inset by 2 px so 2 px strokes are not clipped at the viewBox edges */}
        {box(2, 118, pre ? 'Masked image' : 'Full image', pre ? '25% of 196 patches' : '196 patches', true)}
        {arrow(120, 150)}
        {box(150, 165, 'ViT-B/16 encoder', pre ? 'trained' : 'initialised from USF-MAE', true, 'var(--m-usfmae)')}
        {arrow(315, 345)}
        {box(345, 150, pre ? 'Decoder (8×512)' : 'Linear head (768→6)', pre ? 'mask tokens + positions' : 'new, random init', true, pre ? 'var(--ink-2)' : 'var(--m-densenet)')}
        {arrow(495, 525)}
        {box(525, 193, pre ? 'Pixels of masked patches' : 'Plane probabilities', pre ? 'MSE on masked only' : 'cross-entropy vs label', true)}
        <text x={W / 2} y={150} textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{pre ? 'No labels used. Learns what ultrasound images look like.' : 'Labels used. Learns which representation directions separate the six planes.'}</text>
      </svg>
    </Figure>
  );
}
