import { useMemo, useState } from 'react';
import { HBars } from '../components/charts';
import { Figure, Seg, Slider, fmtInt, pct } from '../components/ui';
import { useArtifacts } from '../data/artifacts';
import { CLASS_SHORT, FETAL_PLANES_LITERATURE as LIT } from '../data/literature';
import { balancedWeights, effectiveNumberWeights } from '../lib/metrics';
import { expectedLeakShare, simulateLeakage } from '../lib/toyModels';
import { mulberry32 } from '../lib/rng';

type Scheme = 'none' | 'inverse' | 'sqrt' | 'effective';

export function ClassBalance() {
  const { datasetStats } = useArtifacts();
  const verified = !!datasetStats;
  const counts = verified ? datasetStats!.class_counts.train.map((v, i) => v + datasetStats!.class_counts.test[i]) : LIT.classCounts;
  const trainCounts = verified ? datasetStats!.class_counts.train : null;
  const [scheme, setScheme] = useState<Scheme>('inverse');
  const [beta, setBeta] = useState(0.999);
  const basis = trainCounts ?? counts;
  const weights = useMemo(() => {
    if (scheme === 'none') return basis.map(() => 1);
    if (scheme === 'inverse') return balancedWeights(basis);
    if (scheme === 'sqrt') { const w = balancedWeights(basis).map(Math.sqrt); const s = w.reduce((a, b) => a + b, 0); return w.map((v) => (v * w.length) / s); }
    return effectiveNumberWeights(basis, beta);
  }, [scheme, beta, basis]);
  const N = basis.reduce((a, b) => a + b, 0);
  const share = basis.map((n, i) => n * weights[i]);
  const shareSum = share.reduce((a, b) => a + b, 0);
  return (
    <Figure title="Class distribution and loss weighting" prov={verified ? ['model', 'live'] : 'live'}
      provDetail={verified ? `dataset_stats.json · ${datasetStats!.provenance.script}` : LIT.source}
      caption={<>{verified ? <>Counts computed from <code>FETAL_PLANES_DB_data.csv</code> by notebook 01.</> : <>Counts as reported by {LIT.source}; {LIT.status}</>} The right panel is exact arithmetic: it shows the fraction of the total training loss each class would contribute (at equal per-sample loss) under the chosen weighting.</>}>
      <div className="grid-2">
        <div>
          <b className="small">Images per class{verified ? ' (train + test)' : ''}</b>
          <HBars items={CLASS_SHORT.map((c, i) => ({ label: c, value: counts[i] }))} format={(v) => fmtInt(v)} />
          <p className="small muted">Largest / smallest class ratio: {(Math.max(...counts) / Math.min(...counts)).toFixed(1)}×. “Other” is a heterogeneous catch-all, not a single anatomy.</p>
        </div>
        <div>
          <div className="row" style={{ marginBottom: '0.4rem' }}>
            <Seg label="Weighting" value={scheme} onChange={setScheme} options={[
              { value: 'none', label: 'None' }, { value: 'inverse', label: 'Inverse freq.' }, { value: 'sqrt', label: '√ inverse' }, { value: 'effective', label: 'Effective number' },
            ]} />
          </div>
          {scheme === 'effective' && <Slider label="β" value={beta} min={0.9} max={0.9999} step={0.0001} onChange={setBeta} format={(v) => v.toFixed(4)} />}
          <b className="small">Share of training loss</b>
          <HBars items={CLASS_SHORT.map((c, i) => ({ label: c, value: share[i] / shareSum, sub: `w=${weights[i].toFixed(2)}` }))} format={(v) => pct(v)} max={0.6} barColor="var(--m-densenet)" />
          <p className="small muted">Unweighted, “Other” supplies {pct(basis[5] / N)} of the gradient signal and the abdomen {pct(basis[0] / N)}.</p>
        </div>
      </div>
    </Figure>
  );
}

/** Illustrative images-per-patient distribution with the literature mean (12,400 / 1,792 ≈ 6.9). */
function illustrativeImagesPerPatient(): number[] {
  const rand = mulberry32(3);
  const out: number[] = [];
  let total = 0;
  for (let i = 0; i < LIT.totalPatients; i++) {
    // shifted geometric with mean ≈ 6.9
    const p = 1 / 5.9;
    let n = 1;
    while (rand() > p && n < 60) n++;
    out.push(n); total += n;
  }
  return out.map((n) => Math.max(1, Math.round((n * LIT.totalImages) / total)));
}

export function LeakageLab() {
  const { datasetStats } = useArtifacts();
  const ipp = useMemo(() => datasetStats?.images_per_patient ?? illustrativeImagesPerPatient(), [datasetStats]);
  const [testFrac, setTestFrac] = useState(0.2);
  const [frames, setFrames] = useState(4);
  const [spread, setSpread] = useState(1.6);
  const [seed, setSeed] = useState(11);
  const sim = useMemo(() => simulateLeakage({ patients: 160, acqPerPatient: 3, framesPerAcq: frames, classSignal: 1.0, acquisitionSpread: spread, frameNoise: 0.15, testFraction: testFrac, seed }), [frames, spread, testFrac, seed]);
  const expected = expectedLeakShare(ipp, testFrac);
  const mean = ipp.reduce((a, b) => a + b, 0) / ipp.length;
  return (
    <Figure title="Image-level versus patient-level splitting" prov={datasetStats ? ['model', 'live', 'illustrative'] : ['live', 'illustrative']}
      caption={<>
        <b>Left</b>: if images are assigned to train/test independently with test fraction <i>f</i>, a test image from a patient with <i>n</i> images has a sibling in training with probability 1 − f<sup>n−1</sup>. This is exact. It uses {datasetStats ? <>the real images-per-patient distribution from the metadata CSV (mean {mean.toFixed(1)})</> : <>an <b>illustrative</b> distribution with the published mean of {mean.toFixed(1)} images per patient (replace by exporting dataset statistics)</>}.{' '}
        <b>Right</b>: an illustrative 1-nearest-neighbour “model” on synthetic features. Each patient contributes three acquisitions, each saved as several near-identical frames (FETAL_PLANES_DB file names such as <code>Patient00001_Plane1_3_of_15</code> indicate saved sequences). The gap between the two accuracies is the leakage.
      </>}>
      <div className="grid-2">
        <div className="stack">
          <Slider label="Test fraction f" value={testFrac} min={0.1} max={0.5} step={0.05} onChange={setTestFrac} format={(v) => pct(v, 0)} />
          <div style={{ fontFamily: 'var(--serif)', fontSize: '2.6rem', lineHeight: 1 }}>{pct(expected, 1)}</div>
          <div className="small">of test images would come from a patient who also has images in the training set under a random image-level split. With a patient-level split the figure is 0% by construction.</div>
          <p className="small muted" style={{ margin: 0 }}>The authors’ split (first 896 patients chronologically for training, next 896 for testing) is patient-disjoint and also temporal. Our validation set is carved patient-disjointly from the training half.</p>
        </div>
        <div className="stack">
          <Slider label="Near-duplicate frames per acquisition" value={frames} min={1} max={10} step={1} onChange={setFrames} />
          <Slider label="Between-acquisition variability" value={spread} min={0.2} max={3} step={0.1} onChange={setSpread} format={(v) => v.toFixed(1)} />
          <HBars items={[
            { label: 'Random image split', value: sim.randomSplitAcc, color: 'var(--p-missing)' },
            { label: 'Patient-level split', value: sim.patientSplitAcc, color: 'var(--p-model)' },
          ]} max={1} format={(v) => pct(v)} />
          <div className="row small"><span>{sim.nImages} synthetic images · {pct(sim.leakedShare, 0)} of random-split test images have a same-patient sibling in train</span>
            <button className="btn" onClick={() => setSeed((s) => s + 1)}>Resample</button></div>
        </div>
      </div>
    </Figure>
  );
}
