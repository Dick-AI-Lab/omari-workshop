import { useEffect, useMemo, useState } from 'react';
import { ImagePicker } from '../components/ImagePicker';
import { Empty, Figure, GrayCanvas, Seg, Slider, fmtInt, fmtM } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { useImageSource } from '../data/imageSource';
import type { FeatureLayer } from '../data/types';
import { DENSENET161, densenet161Stages, denseInputs, totalParams } from '../lib/architectures';
import { Gray, conv2d, relu, resize, toDisplay } from '../lib/imageOps';

const PRESETS: Record<string, number[][]> = {
  Identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  'Sobel x': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  'Sobel y': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  Laplacian: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
  'Box blur': [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  Sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  'Diagonal edge': [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
};

export function ConvExplorer() {
  const src = useImageSource();
  const [preset, setPreset] = useState('Sobel x');
  const [k, setK] = useState<number[][]>(PRESETS['Sobel x']);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(1);
  const [dilation, setDilation] = useState(1);
  const [applyRelu, setApplyRelu] = useState(true);
  const input = useMemo(() => resize(src.gray, 160, Math.round(160 * (src.gray.h / src.gray.w)), 'area'), [src.gray]);
  const raw = useMemo(() => conv2d(input, { kernel: k, stride, padding, dilation }), [input, k, stride, padding, dilation]);
  const out = useMemo(() => (applyRelu ? toDisplay(relu(raw)) : toDisplay(raw, true)), [raw, applyRelu]);
  const set = (i: number, j: number, v: number) => { const n = k.map((r) => r.slice()); n[i][j] = v; setK(n); setPreset('Custom'); };
  const eff = dilation * 2 + 1;
  return (
    <Figure title="A single learned kernel, by hand" prov="live"
      caption={<>A 3×3 cross-correlation over the selected image, computed in your browser. Edit the nine weights or pick a preset. With ReLU off the map is shown signed (grey = 0); with ReLU on, only positive responses survive, as after a convolution–ReLU pair in a CNN. Output size follows <i>⌊(H + 2p − d(k−1) − 1)/s⌋ + 1</i>: currently {raw.w}×{raw.h} from {input.w}×{input.h}.</>}>
      <ImagePicker compact />
      <div className="grid-3">
        <GrayCanvas img={input} label={`input ${input.w}×${input.h}`} />
        <div className="stack">
          <Seg label="Preset" value={preset} onChange={(p) => { setPreset(p); if (PRESETS[p]) setK(PRESETS[p]); }} options={Object.keys(PRESETS).map((p) => ({ value: p, label: p }))} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 4.2rem)', gap: 4 }} role="group" aria-label="kernel weights">
            {k.map((row, i) => row.map((v, j) => (
              <input key={`${i}${j}`} type="number" step={0.25} value={Number(v.toFixed(3))} aria-label={`weight row ${i + 1} column ${j + 1}`}
                onChange={(e) => set(i, j, Number(e.target.value))}
                style={{ background: v > 0 ? `rgba(0,114,178,${Math.min(0.5, Math.abs(v) / 4)})` : v < 0 ? `rgba(213,94,0,${Math.min(0.5, Math.abs(v) / 4)})` : undefined }} />
            )))}
          </div>
          <Slider label="Stride s" value={stride} min={1} max={4} onChange={setStride} />
          <Slider label="Padding p" value={padding} min={0} max={4} onChange={setPadding} />
          <Slider label="Dilation d" value={dilation} min={1} max={4} onChange={setDilation} format={(v) => `${v} (span ${v * 2 + 1}px)`} />
          <label className="row small"><input type="checkbox" checked={applyRelu} onChange={(e) => setApplyRelu(e.target.checked)} /> Apply ReLU</label>
          <span className="small muted">One output unit sees a {eff}×{eff} input window.</span>
        </div>
        <GrayCanvas img={out} pixelated={stride > 1} label={`feature map ${raw.w}×${raw.h}`} />
      </div>
    </Figure>
  );
}

export function ReceptiveFieldFigure() {
  const src = useImageSource();
  const stages = useMemo(() => densenet161Stages(6), []);
  const [si, setSi] = useState(4);
  const [pt, setPt] = useState<[number, number]>([112, 112]);
  const img = useMemo(() => resize(src.gray, 224, 224, 'bilinear'), [src.gray]);
  const s = stages[si];
  const marked = useMemo(() => {
    const o: Gray = { w: 224, h: 224, data: new Float32Array(img.data) };
    const half = s.rf / 2;
    // snap to the centre of the output unit containing the clicked point
    const cx = Math.floor(pt[0] / s.jump) * s.jump + s.jump / 2;
    const cy = Math.floor(pt[1] / s.jump) * s.jump + s.jump / 2;
    for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) {
      const inside = Math.abs(x - cx) <= half && Math.abs(y - cy) <= half;
      const unit = Math.abs(x - cx) < s.jump / 2 && Math.abs(y - cy) < s.jump / 2;
      if (!inside) o.data[y * 224 + x] *= 0.35;
      if (unit) o.data[y * 224 + x] = 1;
    }
    return o;
  }, [img, s, pt]);
  const total = totalParams(stages);
  return (
    <Figure title="How far does one DenseNet-161 unit see?" prov="live"
      caption={<>Exact theoretical receptive field computed from the DenseNet-161 definition (torchvision): each 3×3 convolution adds 2·<i>J</i> pixels, where <i>J</i> is the cumulative stride. Click the image to move the unit (white square = one output position). By denseblock 3 the theoretical field exceeds the 224-pixel input. The <i>effective</i> receptive field, where gradients are concentrated, is much smaller and roughly Gaussian (Luo et al., 2016).</>}>
      <div className="split-wide">
        <GrayCanvas img={marked} onClick={(x, y) => setPt([x, y])} label={`${s.name}: RF ${s.rf}px, stride ${s.jump}`} />
        <div>
          <table className="data">
            <thead><tr><th>Stage</th><th className="num">Output</th><th className="num">Channels</th><th className="num">RF (px)</th><th className="num">Params</th></tr></thead>
            <tbody>
              {stages.map((st, i) => (
                <tr key={st.name} onClick={() => setSi(i)} style={{ cursor: 'pointer', background: i === si ? '#e8eef1' : undefined }}>
                  <td>{st.name}</td><td className="num">{st.size}×{st.size}</td><td className="num">{fmtInt(st.channels)}</td><td className="num">{st.rf}</td><td className="num">{fmtM(st.params)}</td>
                </tr>
              ))}
              <tr><td colSpan={4}><b>Total (6-class head)</b></td><td className="num"><b>{fmtM(total)}</b></td></tr>
            </tbody>
          </table>
          <p className="small muted">With the original 1,000-class head the same arithmetic gives {fmtM(totalParams(densenet161Stages(1000)))}, matching torchvision’s published 28.7 M.</p>
        </div>
      </div>
    </Figure>
  );
}

export function DenseBlockFigure() {
  const [layers, setLayers] = useState(4);
  const [k, setK] = useState<number>(DENSENET161.growth as number);
  const [hover, setHover] = useState<number | null>(null);
  const c0 = 96;
  const ins = denseInputs(c0, layers, k);
  const W = 640, H = 220, x0 = 60, dx = (W - 2 * x0) / layers;
  return (
    <Figure title="Dense connectivity: every layer reads everything before it" prov="live"
      caption={<>Layer ℓ receives the channel-wise concatenation of the block input and all ℓ−1 earlier outputs, and contributes <i>k</i> new channels (growth rate <i>k</i> = 48 in DenseNet-161). Hover a layer to see what it reads. Channel counts are exact; arc thickness is proportional to the number of channels carried. Compare a residual block, which <i>adds</i> features and keeps the width fixed.</>}>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <div style={{ width: '14rem' }}><Slider label="Layers in block" value={layers} min={1} max={8} onChange={setLayers} /></div>
        <Seg label="Growth rate k" value={k} onChange={setK} options={[12, 32, 48].map((v) => ({ value: v, label: String(v) }))} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Dense block connectivity">
        {Array.from({ length: layers + 1 }, (_, src) => Array.from({ length: layers + 1 }, (_, dst) => {
          if (dst <= src) return null;
          const xs = x0 + src * dx, xd = x0 + dst * dx;
          const active = hover == null || hover === dst;
          const width = src === 0 ? c0 / 24 : k / 24;
          return <path key={`${src}-${dst}`} d={`M${xs},120 C${xs},${120 - (dst - src) * 22} ${xd},${120 - (dst - src) * 22} ${xd},120`} fill="none"
            stroke={active ? 'var(--m-densenet)' : 'var(--rule)'} strokeOpacity={active ? 0.75 : 1} strokeWidth={width} />;
        }))}
        {ins.map((c, l) => (
          <g key={l} transform={`translate(${x0 + l * dx}, 120)`} onMouseEnter={() => setHover(l)} onMouseLeave={() => setHover(null)} style={{ cursor: 'default' }}>
            <rect x={-26} y={0} width={52} height={36} rx={4} fill={l === 0 ? 'var(--well)' : hover === l ? 'var(--m-densenet)' : '#fff'} stroke="var(--ink-2)" />
            <text x={0} y={22} textAnchor="middle" style={{ fill: l === 0 || hover === l ? '#fff' : 'var(--ink)', fontWeight: 600 }}>{l === 0 ? 'x₀' : `H${l}`}</text>
            <text x={0} y={54} textAnchor="middle">{l === 0 ? `${c0} ch` : `reads ${c}`}</text>
            <text x={0} y={68} textAnchor="middle" className="axis-title">{l === 0 ? '' : `writes ${k}`}</text>
          </g>
        ))}
        <text x={W - 10} y={200} textAnchor="end" style={{ fill: 'var(--ink)' }}>block output: {c0 + layers * k} channels</text>
      </svg>
    </Figure>
  );
}

function useFeatureTiles(layer?: FeatureLayer) {
  const [tiles, setTiles] = useState<Gray[]>([]);
  useEffect(() => {
    let c = false;
    setTiles([]);
    if (!layer) return;
    Promise.all(layer.channels.map((ch) => loadGrayPNG(ch.png))).then((t) => !c && setTiles(t)).catch(() => {});
    return () => { c = true; };
  }, [layer]);
  return tiles;
}

export function DenseNetActivations() {
  const src = useImageSource();
  const { manifest } = useArtifacts();
  const layers = src.sample?.features?.densenet161 ?? [];
  const [li, setLi] = useState(0);
  const layer = layers[Math.min(li, layers.length - 1)];
  const tiles = useFeatureTiles(layer);
  const model = manifest?.models.find((m) => m.id === 'densenet161');
  if (!layers.length) {
    return (
      <Figure title="Real DenseNet-161 feature maps" prov="missing" caption="Feature maps are taken from the fine-tuned checkpoint with forward hooks on the outputs of each dense block. They are never simulated.">
        <ImagePicker compact />
        <Empty title={src.kind === 'sample' ? 'No feature maps exported for this sample' : 'Select an exported FETAL_PLANES_DB sample'}>
          Export with <code>python scripts/export_website_artifacts.py --features densenet161</code>. The exporter records the layer name, tensor shape and the rule used to choose channels.
        </Empty>
      </Figure>
    );
  }
  return (
    <Figure title="Real DenseNet-161 feature maps" prov="model" provDetail={`${model?.checkpoint_id ?? ''}`}
      caption={<>Channels from <code>{layer.layer}</code> (tensor {layer.shape.join('×')}), selected by: {layer.selection}. Each tile is min–max scaled independently. Early layers respond to edges and speckle texture; deeper layers are coarser and more class-selective, but individual channels are rarely interpretable on their own.</>}>
      <ImagePicker compact />
      <Seg label="Layer" value={li} onChange={setLi} options={layers.map((l, i) => ({ value: i, label: l.layer.replace('features.', '') }))} />
      <div className="grid-4" style={{ marginTop: '0.8rem', gridTemplateColumns: 'repeat(auto-fill, minmax(7rem, 1fr))' }}>
        {tiles.map((t, i) => <GrayCanvas key={i} img={t} cmap="inferno" pixelated label={`#${layer.channels[i].index}`} />)}
      </div>
    </Figure>
  );
}
