import { useEffect, useMemo, useState } from 'react';
import { ChapterHead } from '../components/shell';
import { Empty, Figure, GrayCanvas, ModelTag } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { CLASS_SHORT } from '../data/literature';
import type { SampleInfo } from '../data/types';
import { Gray } from '../lib/imageOps';
import { PredictionBars } from '../visualizations/PredictionBars';

function useSampleImage(s?: SampleInfo) {
  const [g, setG] = useState<Gray | null>(null);
  useEffect(() => { let c = false; setG(null); if (s) loadGrayPNG(s.image).then((x) => !c && setG(x)).catch(() => {}); return () => { c = true; }; }, [s]);
  return g;
}

export function OpeningChallenge() {
  const { manifest, status } = useArtifacts();
  const samples = manifest?.samples ?? [];
  const pool = useMemo(() => {
    const opening = samples.filter((s) => s.tags?.includes('opening'));
    const dis = samples.filter((s) => s.tags?.includes('disagreement'));
    return [...opening, ...dis.filter((d) => !opening.includes(d))];
  }, [samples]);
  const [idx, setIdx] = useState(0);
  const [guess, setGuess] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false);
  const sample = pool[idx] ?? samples[0];
  const img = useSampleImage(sample);
  const hasPreds = sample && Object.keys(sample.predictions ?? {}).length > 0;

  const next = (d: number) => { setIdx((i) => (pool.length ? (i + d + pool.length) % pool.length : 0)); setGuess(null); setRevealed(false); };

  if (!sample) {
    return (
      <Figure title="Which plane is this?" prov="missing" caption={<>The opening challenge uses a real FETAL_PLANES_DB test image and the three models' actual predictions. <b>Nothing is shown until those are exported</b>, because a stand-in image or invented probabilities would undermine the point of the exercise.</>}>
        <Empty title={status === 'loading' ? 'Looking for the artifact bundle…' : 'No exported samples yet'}>
          Run <code>python scripts/export_website_artifacts.py --config configs/export.yaml</code> after training (notebooks 02–04), or copy a bundle into <code>website/public/experiments/</code>. Mark one sample with the <code>opening</code> tag to choose the image used here.
        </Empty>
      </Figure>
    );
  }

  return (
    <Figure title="Which plane is this?" prov={hasPreds ? 'model' : 'missing'}
      provDetail={manifest?.models.map((m) => `${m.display_name}: ${m.checkpoint_id}`).join('\n')}
      caption={<>A held-out image from the <b>{sample.split}</b> split ({sample.machine ?? 'machine not recorded'}). Choose a plane first; the reveal shows the expert label and the softmax outputs of each fine-tuned model. Use the arrows to step through images where the models disagree.</>}>
      <div className="split">
        <div>
          {img ? <GrayCanvas img={img} ariaLabel="Fetal ultrasound image from FETAL_PLANES_DB" /> : <div className="well" style={{ aspectRatio: '4/3' }} />}
          <div className="row" style={{ marginTop: '0.6rem', justifyContent: 'space-between' }}>
            <button className="btn" onClick={() => next(-1)} disabled={pool.length < 2}>Previous image</button>
            <span className="muted small">{pool.length ? `${idx + 1} of ${pool.length}` : ''}{sample.tags?.includes('disagreement') ? ' · models disagree' : ''}</span>
            <button className="btn" onClick={() => next(1)} disabled={pool.length < 2}>Next image</button>
          </div>
        </div>
        <div className="stack">
          <div className="control-label"><span>Your answer</span></div>
          <div className="ask-options" role="group" aria-label="Choose the anatomical plane">
            {CLASS_SHORT.map((c, i) => (
              <button key={c} aria-pressed={guess === i} onClick={() => setGuess(i)}>{c}</button>
            ))}
          </div>
          <button className="btn primary" style={{ justifySelf: 'start' }} disabled={guess == null && !revealed} onClick={() => setRevealed(!revealed)}>
            {revealed ? 'Hide models' : 'Reveal label and models'}
          </button>
          {revealed && (
            <>
              <p className="small" style={{ margin: 0 }}>
                Expert label: <b>{CLASS_SHORT[sample.label]}</b>{sample.brain_plane && sample.label === 1 ? ` (${sample.brain_plane})` : ''}.{' '}
                {guess != null && (guess === sample.label ? 'You matched the expert.' : `You chose ${CLASS_SHORT[guess]}.`)}
              </p>
              {hasPreds ? <PredictionBars preds={sample.predictions} truth={sample.label} highlight={guess ?? undefined} /> : (
                <Empty title="No predictions exported for this sample">Export predictions with the artifact pipeline.</Empty>
              )}
            </>
          )}
        </div>
      </div>
    </Figure>
  );
}

export default function Ch1Opening() {
  return (
    <section className="chapter" id="opening">
      <ChapterHead id="opening" />
      <p className="lede">A sonographer saves a frozen frame. Before any measurement, someone must decide what it shows: a fetal brain, abdomen, femur or thorax, the maternal cervix, or something else.</p>
      <p>
        That decision is routine for an expert and is the task of this workshop: six-way classification of maternal–fetal ultrasound planes on FETAL_PLANES_DB, 12,400 images from 1,792 patients acquired at two Barcelona hospitals. We will follow three models that learn their image representations in very different ways: a supervised convolutional network (<ModelTag id="densenet161" />), a supervised vision transformer (<ModelTag id="vit_b16" />), and an ultrasound-specific self-supervised foundation model (<ModelTag id="usf_mae" />).
      </p>
      <OpeningChallenge />
      <p>
        Three different quantities are hiding in that reveal. <b>Correctness</b> asks whether the top class matches the expert. <b>Confidence</b> is how much probability mass the model puts there, which is only meaningful if the model is calibrated. <b>Reasoning</b> concerns which image evidence produced the output: the anatomy, the depth setting, the burned-in text, or the shape of the field of view. Accuracy measures the first, calibration the second. The third has no single metric, and most of this workshop is about why.
      </p>
      <p className="pull-question" data-beat>If all three models predict the correct plane, do we know they learned the right features?</p>
    </section>
  );
}
