import { useEffect, useState } from 'react';
import { useArtifacts } from '../data/artifacts';
import { useImageSource } from '../data/imageSource';
import type { ModelId, PerturbationEntry, Prediction } from '../data/types';
import { Gray } from '../lib/imageOps';
import { Empty, Prov } from '../components/ui';
import { PredictionBars } from './PredictionBars';

function grayToPngBase64(g: Gray): string {
  const c = document.createElement('canvas');
  c.width = g.w; c.height = g.h;
  const ctx = c.getContext('2d')!;
  const id = ctx.createImageData(g.w, g.h);
  for (let i = 0, j = 0; i < g.data.length; i++, j += 4) { const v = Math.round(g.data[i] * 255); id.data[j] = v; id.data[j + 1] = v; id.data[j + 2] = v; id.data[j + 3] = 255; }
  ctx.putImageData(id, 0, 0);
  return c.toDataURL('image/png').split(',')[1];
}

export async function liveInference(url: string, img: Gray, models: ModelId[]): Promise<Partial<Record<ModelId, Prediction>>> {
  const r = await fetch(url + '/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_png_base64: grayToPngBase64(img), models }) });
  if (!r.ok) throw new Error(`backend answered ${r.status}`);
  const j = await r.json();
  return j.predictions;
}

/**
 * Shows model predictions for a transformed image ONLY when they are real:
 *  (a) a precomputed perturbation with exactly these settings exists for the selected sample, or
 *  (b) the local inference server is running and the participant asks for live inference.
 * Otherwise it says why nothing is shown.
 */
export function InferenceReadout({ match, transformed, isIdentity, onApply }: { match?: PerturbationEntry; transformed: Gray; isIdentity: boolean; onApply?: (e: PerturbationEntry) => void }) {
  const { backend } = useArtifacts();
  const src = useImageSource();
  const [live, setLive] = useState<{ preds: Partial<Record<ModelId, Prediction>>; err?: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const base = src.sample?.predictions;
  // a live result belongs to one exact image; discard it as soon as the image or settings change
  useEffect(() => { setLive(null); }, [transformed]);

  const run = async () => {
    setBusy(true);
    try { setLive({ preds: await liveInference(backend.url, transformed, backend.models) }); }
    catch (e) { setLive({ preds: {}, err: String(e) }); }
    finally { setBusy(false); }
  };

  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <b className="small">Model predictions for this image</b>
        {isIdentity && base ? <Prov kind="model" /> : match ? <Prov kind="model" detail={`${match.provenance.script} · ${match.label}`} /> : live && !live.err ? <Prov kind="inference" /> : <Prov kind="missing" />}
      </div>
      {isIdentity && base && Object.keys(base).length ? <PredictionBars preds={base} truth={src.sample?.label} /> :
        match ? <><PredictionBars preds={match.predictions} truth={src.sample?.label} /><span className="muted small">Precomputed for “{match.label}”. Stochastic perturbations used the exported noise realisation, not the one drawn in your browser.</span></> :
        live && !live.err ? <PredictionBars preds={live.preds} truth={src.sample?.label} /> :
        (
          <Empty title={src.kind === 'phantom' ? 'Models are not run on the phantom' : 'No verified prediction for these settings'}>
            {src.kind === 'phantom'
              ? 'The phantom is a synthetic test object, so no plane label exists. Select an exported FETAL_PLANES_DB sample.'
              : 'Precomputed predictions exist only for the perturbation grid exported by the pipeline. '}
            {onApply && (src.sample?.perturbations?.length ?? 0) > 0 && (
              <div className="row" style={{ gap: '0.35rem', marginTop: '0.45rem' }}>
                <span className="small">Exported settings:</span>
                {src.sample!.perturbations!.filter((e) => ['brightness', 'contrast', 'gamma', 'resolution', 'speckle'].includes(e.type)).map((e) => (
                  <button key={e.type + e.param} className="btn" style={{ padding: '0.1rem 0.45rem' }} onClick={() => onApply(e)}>{e.label}</button>
                ))}
              </div>
            )}
            {backend.ok && src.kind !== 'phantom' && (
              <div style={{ marginTop: '0.5rem' }}>
                <button className="btn primary" onClick={run} disabled={busy}>{busy ? 'Running…' : `Run live inference (${backend.url})`}</button>
                {live?.err && <div className="small" style={{ color: 'var(--p-missing)' }}>{live.err}</div>}
              </div>
            )}
            {!backend.ok && src.kind !== 'phantom' && <div className="small muted" style={{ marginTop: '0.4rem' }}>For any setting, start the local server: <code>python -m omari_us.serve --config configs/export.yaml</code></div>}
          </Empty>
        )}
    </div>
  );
}
