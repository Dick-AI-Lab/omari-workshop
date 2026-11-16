import { useEffect, useMemo, useState } from 'react';
import { ChapterHead } from '../components/shell';
import { Empty, FileError, Figure, GrayCanvas, MODEL_META, ModelTag, Seg, pct } from '../components/ui';
import { loadGrayPNG, useArtifacts } from '../data/artifacts';
import { CLASS_SHORT } from '../data/literature';
import type { ModelId, SampleInfo } from '../data/types';
import { Gray } from '../lib/imageOps';
import { PredictionBars } from '../visualizations/PredictionBars';

function useSampleImage(s?: SampleInfo) {
  const [g, setG] = useState<Gray | null>(null);
  useEffect(() => { let c = false; setG(null); if (s) loadGrayPNG(s.image).then((x) => !c && setG(x)).catch((e) => console.error('[opening] image', s.image, e)); return () => { c = true; }; }, [s]);
  return g;
}

type PoolKey = 'workshop' | 'disagree' | 'agree' | 'wrong' | 'all';

const predsOf = (s: SampleInfo) => (Object.keys(MODEL_META) as ModelId[]).map((m) => s.predictions?.[m]).filter(Boolean) as { pred: number; probs: number[] }[];
const disagrees = (s: SampleInfo) => new Set(predsOf(s).map((p) => p.pred)).size > 1;
const anyWrong = (s: SampleInfo) => predsOf(s).some((p) => p.pred !== s.label);

export function OpeningChallenge({ closing = false }: { closing?: boolean }) {
  const { manifest, status, error } = useArtifacts();
  const samples = manifest?.samples ?? [];
  const [poolKey, setPoolKey] = useState<PoolKey>('workshop');
  const pools = useMemo(() => {
    const opening = samples.filter((s) => s.tags?.includes('opening'));
    const dis = samples.filter((s) => s.tags?.includes('disagreement') || disagrees(s));
    const openingFirst = (xs: SampleInfo[]) => [...opening.filter((o) => xs.includes(o)), ...xs.filter((x) => !opening.includes(x))];
    return {
      workshop: [...opening, ...dis.filter((d) => !opening.includes(d))],
      disagree: openingFirst(samples.filter(disagrees)),
      agree: openingFirst(samples.filter((s) => predsOf(s).length > 1 && !disagrees(s))),
      wrong: openingFirst(samples.filter(anyWrong)),
      all: openingFirst(samples),
    } as Record<PoolKey, SampleInfo[]>;
  }, [samples]);
  const pool = pools[poolKey];
  const [idx, setIdx] = useState(0);
  const [guess, setGuess] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false);
  const sample = pool[idx] ?? pool[0] ?? samples[0];
  const img = useSampleImage(sample);
  const hasPreds = !!sample && Object.keys(sample.predictions ?? {}).length > 0;
  const models = (Object.keys(MODEL_META) as ModelId[]).filter((m) => sample?.predictions?.[m]);

  const next = (d: number) => { setIdx((i) => (pool.length ? (i + d + pool.length) % pool.length : 0)); setGuess(null); setRevealed(false); };
  const choosePool = (k: PoolKey) => { setPoolKey(k); setIdx(0); setGuess(null); setRevealed(false); };

  if (status === 'error') {
    return <Figure title="Which plane is this?" prov="error"><FileError message={error} /></Figure>;
  }
  if (!sample) {
    return (
      <Figure title="Which plane is this?" prov="missing" caption={<>The opening challenge uses a real FETAL_PLANES_DB test image and the three models' actual predictions. <b>Nothing is shown until those are exported</b>, because a stand-in image or invented probabilities would undermine the point of the exercise.</>}>
        <Empty title={status === 'loading' ? 'Looking for the artifact bundle…' : 'No exported samples yet'}>
          Build the bundle with <code>python scripts/finalize_website_bundle.py --artifacts content/omari_artifacts/omari_artifacts --out website/public/experiments</code> so that <code>website/public/experiments/manifest.json</code> exists. The sample tagged <code>opening</code> is the image used here.
        </Empty>
      </Figure>
    );
  }

  const poolOptions: { value: PoolKey; label: string }[] = [
    { value: 'workshop', label: `Workshop set (${pools.workshop.length})` },
    { value: 'disagree', label: `Models disagree (${pools.disagree.length})` },
    { value: 'agree', label: `All agree (${pools.agree.length})` },
    { value: 'wrong', label: `A model is wrong (${pools.wrong.length})` },
    { value: 'all', label: `All exported (${pools.all.length})` },
  ];
  const provDetail = [
    ...(manifest?.models.map((m) => `${m.display_name}: ${m.checkpoint_id}`) ?? []),
    manifest?.sample_predictions_source ? `Predictions: ${manifest.sample_predictions_source}` : '',
  ].filter(Boolean).join('\n');

  return (
    <Figure title={closing ? 'The opening image, again' : 'Which plane is this?'} prov={hasPreds ? 'model' : 'missing'} provDetail={provDetail}
      caption={<>A held-out image from the <b>{sample.split}</b> split ({sample.machine ?? 'machine not recorded'}, {sample.patient}). Choose a plane first; the reveal shows the expert label and the softmax output of each fine-tuned model (100% labels, seed 0).{' '}
        {manifest?.sample_predictions_source
          ? <>These are the saved test-set predictions behind the Chapter 4 metrics, not a separate re-run (full source in the badge and in the Chapter 7 provenance notes).</>
          : <>In this bundle the probabilities were recomputed by the exporter on the exported image, so in rare cases they can differ slightly from the evaluation run in Chapter 4.</>}{' '}
        Switch sets to compare images where the models agree with images where they disagree.{closing ? ' Has your reading of these probabilities changed since the start?' : ''}</>}>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <Seg label="Images" value={poolKey} onChange={choosePool} options={poolOptions.map((o) => ({ ...o, disabled: !pools[o.value].length }))} />
      </div>
      <div className="split">
        <div>
          {img ? <GrayCanvas img={img} ariaLabel="Fetal ultrasound image from FETAL_PLANES_DB" /> : <div className="well" style={{ aspectRatio: '4/3' }} />}
          <div className="row" style={{ marginTop: '0.6rem', justifyContent: 'space-between' }}>
            <button className="btn" onClick={() => next(-1)} disabled={pool.length < 2}>Previous image</button>
            <span className="muted small">{pool.length ? `${Math.min(idx, pool.length - 1) + 1} of ${pool.length}` : ''}{sample.tags?.includes('opening') ? ' · opening image' : ''}{revealed && disagrees(sample) ? ' · models disagree' : ''}</span>
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
              {hasPreds ? <>
                <PredictionBars preds={sample.predictions} truth={sample.label} highlight={guess ?? undefined} />
                <ul className="small" style={{ margin: 0, paddingLeft: '1.1rem', fontFamily: 'var(--sans)' }}>
                  {models.map((m) => {
                    const p = sample.predictions[m]!;
                    const ok = p.pred === sample.label;
                    return <li key={m}><ModelTag id={m} />: {CLASS_SHORT[p.pred]}, p = {pct(p.probs[p.pred])} {ok ? '(correct)' : <b style={{ color: 'var(--p-missing)' }}>(wrong; p(expert label) = {pct(p.probs[sample.label])})</b>}</li>;
                  })}
                </ul>
              </> : (
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
        That decision is routine for an expert and is the task of this workshop: six-way classification of maternal–fetal ultrasound planes on FETAL_PLANES_DB, 12,400 images from 1,792 patients acquired at two Barcelona hospitals. We will follow three models that learn their image representations in very different ways: a supervised convolutional network (<ModelTag id="densenet161" />), a supervised vision transformer (<ModelTag id="vit_b16" />), and an ultrasound-specific self-supervised foundation model (<ModelTag id="usf_mae" />, fine-tuned from its 500-epoch pretrained checkpoint).
      </p>
      <OpeningChallenge />
      <p>
        Three different quantities are hiding in that reveal. <b>Correctness</b> asks whether the top class matches the expert. <b>Confidence</b> is how much probability mass the model puts there, which is only meaningful if the model is calibrated. <b>Reasoning</b> concerns which image evidence produced the output: the anatomy, the depth setting, the burned-in text, or the shape of the field of view. Accuracy measures the first, calibration the second. The third has no single metric, and most of this workshop is about why.
      </p>
      <p className="pull-question" data-beat>If all three models predict the correct plane, do we know they learned the right features?</p>
    </section>
  );
}
