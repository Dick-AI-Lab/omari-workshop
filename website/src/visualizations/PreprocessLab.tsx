import { useMemo, useState } from 'react';
import { Histogram } from '../components/charts';
import { ImagePicker } from '../components/ImagePicker';
import { Figure, GrayCanvas, Seg, Slider } from '../components/ui';
import { useImageSource } from '../data/imageSource';
import {
  FitMode, Gray, NormMethod, addSpeckle, adjust, fitToSquare, histogram, normalize, simulateResolution, stats,
} from '../lib/imageOps';
import { InferenceReadout } from './InferenceReadout';

const NORM_LABELS: Record<NormMethod, string> = {
  none: 'None [0,1]', minmax: 'Min–max', zscore: 'Per-image z', imagenet: 'ImageNet μ/σ', half: '0.5 / 0.5', histeq: 'Hist. eq.', clahe: 'CLAHE',
};

function overlayGrid(g: Gray, p: number, content?: { x: number; y: number; w: number; h: number }): Gray {
  const o = { w: g.w, h: g.h, data: new Float32Array(g.data) };
  for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
    if (x % p === 0 || y % p === 0) o.data[y * g.w + x] = o.data[y * g.w + x] * 0.55 + 0.25;
    if (content && (x < content.x || x >= content.x + content.w || y < content.y || y >= content.y + content.h)) o.data[y * g.w + x] *= 0.5;
  }
  return o;
}

export function PreprocessLab() {
  const src = useImageSource();
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [gamma, setGamma] = useState(1);
  const [resFactor, setResFactor] = useState(1);
  const [speckle, setSpeckle] = useState(0);
  const [looks, setLooks] = useState(1);
  const [seed, setSeed] = useState(1);
  const [norm, setNorm] = useState<NormMethod>('imagenet');
  const [fit, setFit] = useState<FitMode>('stretch');
  const [interp, setInterp] = useState<'bilinear' | 'nearest'>('nearest');
  const [showGrid, setShowGrid] = useState(true);

  const sectorMask = useMemo(() => {
    if (src.sectorMask) return src.sectorMask;
    const m = new Uint8Array(src.gray.data.length);
    for (let i = 0; i < m.length; i++) m[i] = src.gray.data[i] > 0.02 ? 1 : 0;
    return m;
  }, [src.gray, src.sectorMask]);

  const processed = useMemo(() => {
    let g = adjust(src.gray, brightness, contrast, gamma);
    g = simulateResolution(g, resFactor);
    if (speckle > 0) g = addSpeckle(g, speckle, looks, seed, sectorMask);
    return g;
  }, [src.gray, brightness, contrast, gamma, resFactor, speckle, looks, seed, sectorMask]);

  const fitted = useMemo(() => fitToSquare(processed, 224, fit, interp), [processed, fit, interp]);
  const normed = useMemo(() => normalize(fitted.out, norm), [fitted, norm]);
  const hIn = useMemo(() => histogram(src.gray, 64), [src.gray]);
  const hOut = useMemo(() => histogram(processed, 64), [processed]);
  const hNet = useMemo(() => histogram({ w: 224, h: 224, data: normed.values }, 64, normed.range[0], normed.range[1]), [normed]);
  const sNet = useMemo(() => stats(normed.values), [normed]);
  const aspect = (src.gray.w / src.gray.h);

  const changed = [brightness !== 0 && 'brightness', contrast !== 1 && 'contrast', gamma !== 1 && 'gamma', resFactor !== 1 && 'resolution', speckle !== 0 && 'speckle'].filter(Boolean) as string[];
  const match = useMemo(() => {
    const ps = src.sample?.perturbations ?? [];
    if (changed.length !== 1) return undefined;
    const t = changed[0];
    const p = t === 'brightness' ? brightness : t === 'contrast' ? contrast : t === 'gamma' ? gamma : t === 'resolution' ? resFactor : speckle;
    return ps.find((e) => e.type === t && Math.abs(e.param - p) < 1e-6);
  }, [src.sample, changed, brightness, contrast, gamma, resFactor, speckle]);

  const reset = () => { setBrightness(0); setContrast(1); setGamma(1); setResFactor(1); setSpeckle(0); setLooks(1); };

  return (
    <Figure title="Preprocessing laboratory" prov="live"
      caption={<>Every control acts in your browser on the selected image. <b>Left</b>: intensity, resolution and speckle changes in image space. <b>Middle</b>: the exact 224×224 tensor a network would see under the chosen geometry and normalisation, with the 16-pixel ViT patch grid. <b>Right</b>: model predictions, shown only when they are verified for these exact settings. The official USF-MAE loader stretches to 224×224 with nearest-neighbour interpolation and ImageNet statistics; that is the default here.</>}>
      <ImagePicker />
      <div className="grid-3" style={{ alignItems: 'start' }}>
        <div className="stack">
          <GrayCanvas img={processed} label={`${src.gray.w}×${src.gray.h} · aspect ${aspect.toFixed(2)}`} />
          <Histogram bins={hOut} compare={hIn} label="intensity (dashed = original)" />
          <div className="controls">
            <Slider label="Brightness" value={brightness} min={-0.4} max={0.4} step={0.02} onChange={setBrightness} format={(v) => (v >= 0 ? '+' : '') + v.toFixed(2)} />
            <Slider label="Contrast (gain about 0.5)" value={contrast} min={0.3} max={2.5} step={0.05} onChange={setContrast} format={(v) => `×${v.toFixed(2)}`} />
            <Slider label="Gamma (display LUT)" value={gamma} min={0.4} max={2.5} step={0.05} onChange={setGamma} format={(v) => v.toFixed(2)} />
            <Slider label="Resolution reduction" value={resFactor} min={1} max={8} step={1} onChange={setResFactor} format={(v) => (v === 1 ? 'native' : `1/${v}`)} />
            <Slider label="Added speckle strength" value={speckle} min={0} max={1} step={0.05} onChange={setSpeckle} format={(v) => v.toFixed(2)} />
            <div className="row">
              <Seg label="Looks L" value={looks} onChange={setLooks} options={[1, 2, 4, 8].map((v) => ({ value: v, label: String(v) }))} />
              <button className="btn" onClick={() => setSeed((s) => s + 1)} disabled={!speckle}>New noise draw</button>
              <button className="btn" onClick={reset}>Reset</button>
            </div>
          </div>
        </div>
        <div className="stack">
          <GrayCanvas img={showGrid ? overlayGrid(normed.display, 16, fit === 'letterbox' ? fitted.content : undefined) : normed.display} pixelated label="network input 224×224" />
          <Histogram bins={hNet} xDomain={normed.range} label={`tensor values · μ ${sNet.mean.toFixed(2)}, σ ${sNet.std.toFixed(2)}`} color="var(--m-densenet)" />
          <div className="controls">
            <Seg label="Geometry" value={fit} onChange={setFit} options={[
              { value: 'stretch', label: 'Stretch', title: 'Resize((224,224)) — distorts aspect ratio' },
              { value: 'letterbox', label: 'Letterbox', title: 'Preserve aspect ratio, pad with zeros' },
              { value: 'centercrop', label: 'Centre crop', title: 'Resize shorter side, crop centre — may cut anatomy' },
            ]} />
            <Seg label="Interpolation" value={interp} onChange={setInterp} options={[{ value: 'nearest', label: 'Nearest' }, { value: 'bilinear', label: 'Bilinear' }]} />
            <Seg label="Normalisation" value={norm} onChange={setNorm} options={(Object.keys(NORM_LABELS) as NormMethod[]).map((k) => ({ value: k, label: NORM_LABELS[k] }))} />
            <label className="row small"><input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} /> Show 16×16 patch grid (14 × 14 = 196 tokens)</label>
            <p className="small muted" style={{ margin: 0 }}>
              {fit === 'stretch' && `Horizontal scale ${fitted.scaleX.toFixed(3)} vs vertical ${fitted.scaleY.toFixed(3)}: circles become ellipses, so head shape cues change with the source aspect ratio.`}
              {fit === 'letterbox' && `Content occupies ${fitted.content.w}×${fitted.content.h} px; ${(100 * (1 - (fitted.content.w * fitted.content.h) / (224 * 224))).toFixed(0)}% of the tensor is padding.`}
              {fit === 'centercrop' && `Keeps ${(100 * (fitted.content.w * fitted.content.h) / (src.gray.w * src.gray.h)).toFixed(0)}% of the source; anatomy near the sector edges can be cut away.`}
            </p>
          </div>
        </div>
        <InferenceReadout match={match} transformed={processed} isIdentity={changed.length === 0} onApply={(e) => {
          reset();
          if (e.type === 'brightness') setBrightness(e.param);
          if (e.type === 'contrast') setContrast(e.param);
          if (e.type === 'gamma') setGamma(e.param);
          if (e.type === 'resolution') setResFactor(e.param);
          if (e.type === 'speckle') setSpeckle(e.param);
        }} />
      </div>
    </Figure>
  );
}
