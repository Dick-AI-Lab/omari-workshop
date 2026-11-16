import { useMemo, useState } from 'react';
import { HBars } from '../components/charts';
import { Empty, FileError, Figure, Seg, Slider, fmtInt, pct } from '../components/ui';
import { useArtifacts } from '../data/artifacts';
import { CLASS_SHORT, FETAL_PLANES_LITERATURE as LIT } from '../data/literature';
import type { DatasetStats } from '../data/types';
import { balancedWeights, effectiveNumberWeights } from '../lib/metrics';
import { expectedLeakShare, simulateLeakage } from '../lib/toyModels';
import { mulberry32 } from '../lib/rng';

type Scheme = 'none' | 'inverse' | 'sqrt' | 'effective';

const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);

/** The bundle's split protocol, minus sentences that only define JSON field names (e.g. 'train_plus_val'). */
function readableProtocol(p?: string): string {
  return (p ?? '').split(/(?<=\.)\s+/).filter((s) => !/'[a-z_]+'/.test(s)).join(' ');
}

/** Split counts in a schema-agnostic way. 1.0: train includes validation; 1.1: train excludes val. */
export function splitCounts(ds: DatasetStats) {
  const cc = ds.class_counts;
  const hasVal = !!cc.val;
  const train = cc.train;
  const val = cc.val ?? null;
  const test = cc.test;
  const trainPlusVal = cc.train_plus_val ?? (val ? train.map((v, i) => v + val[i]) : train);
  return { hasVal, train, val, test, trainPlusVal, total: trainPlusVal.map((v, i) => v + test[i]) };
}

export function ClassBalance() {
  const { datasetStats, files } = useArtifacts();
  const verified = !!datasetStats;
  const sc = verified ? splitCounts(datasetStats!) : null;
  const counts = sc ? sc.total : LIT.classCounts;
  const [scheme, setScheme] = useState<Scheme>('inverse');
  const [beta, setBeta] = useState(0.999);
  // loss weights are computed on the images actually used for gradient steps (train, excluding validation when known)
  const basis = sc ? sc.train : counts;
  const weights = useMemo(() => {
    if (scheme === 'none') return basis.map(() => 1);
    if (scheme === 'inverse') return balancedWeights(basis);
    if (scheme === 'sqrt') { const w = balancedWeights(basis).map(Math.sqrt); const s = w.reduce((a, b) => a + b, 0); return w.map((v) => (v * w.length) / s); }
    return effectiveNumberWeights(basis, beta);
  }, [scheme, beta, basis]);
  const N = sum(basis);
  const share = basis.map((n, i) => n * weights[i]);
  const shareSum = sum(share);
  return (
    <Figure title="Class distribution and loss weighting" prov={verified ? ['data', 'live'] : 'live'}
      provDetail={verified ? `dataset_stats.json · ${datasetStats!.provenance.script}` : LIT.source}
      caption={<>{verified
        ? <>Counts computed by the pipeline from <code>FETAL_PLANES_DB_data.csv</code> and the saved split manifest{sc!.hasVal ? '; training excludes the patient-disjoint validation set' : '; this bundle reports training and validation together'}.</>
        : <>Counts as reported by {LIT.source}; {LIT.status}</>}{' '}
        {files.datasetStats?.state === 'error' && <b style={{ color: '#b00020' }}>dataset_stats.json could not be read ({files.datasetStats.message}); literature values are shown and labelled as such. </b>}
        The right panel is exact arithmetic: the fraction of the total training loss each class would contribute (at equal per-sample loss) under the chosen weighting.</>}>
      <div className="grid-2">
        <div>
          {sc ? (
            <>
              <b className="small">Images per class and split</b>
              <table className="data small" style={{ marginTop: '0.3rem' }}>
                <thead><tr><th>Class</th><th className="num">Train</th>{sc.hasVal && <th className="num">Val</th>}<th className="num">Test</th><th className="num">Total</th></tr></thead>
                <tbody>
                  {CLASS_SHORT.map((c, i) => (
                    <tr key={c}><td>{c}</td><td className="num">{fmtInt(sc.train[i])}</td>{sc.hasVal && <td className="num">{fmtInt(sc.val![i])}</td>}<td className="num">{fmtInt(sc.test[i])}</td><td className="num">{fmtInt(sc.total[i])}</td></tr>
                  ))}
                  <tr><td><b>All</b></td><td className="num"><b>{fmtInt(sum(sc.train))}</b></td>{sc.hasVal && <td className="num"><b>{fmtInt(sum(sc.val!))}</b></td>}<td className="num"><b>{fmtInt(sum(sc.test))}</b></td><td className="num"><b>{fmtInt(sum(sc.total))}</b></td></tr>
                </tbody>
              </table>
              {!sc.hasVal && <p className="small muted">Train column includes the validation patients (schema 1.0 bundle).</p>}
            </>
          ) : (
            <>
              <b className="small">Images per class</b>
              <HBars items={CLASS_SHORT.map((c, i) => ({ label: c, value: counts[i] }))} format={(v) => fmtInt(v)} />
            </>
          )}
          <p className="small muted">Largest / smallest class ratio: {(Math.max(...counts) / Math.min(...counts)).toFixed(1)}×. “Other” is a heterogeneous catch-all, not a single anatomy.</p>
        </div>
        <div>
          <div className="row" style={{ marginBottom: '0.4rem' }}>
            <Seg label="Weighting" value={scheme} onChange={setScheme} options={[
              { value: 'none', label: 'None' }, { value: 'inverse', label: 'Inverse freq.' }, { value: 'sqrt', label: '√ inverse' }, { value: 'effective', label: 'Effective number' },
            ]} />
          </div>
          {scheme === 'effective' && <Slider label="β" value={beta} min={0.9} max={0.9999} step={0.0001} onChange={setBeta} format={(v) => v.toFixed(4)} />}
          <b className="small">Share of training loss{sc ? ' (training images)' : ''}</b>
          <HBars items={CLASS_SHORT.map((c, i) => ({ label: c, value: share[i] / shareSum, sub: `w=${weights[i].toFixed(2)}` }))} format={(v) => pct(v)} max={0.6} barColor="var(--m-densenet)" />
          <p className="small muted">Unweighted, “Other” supplies {pct(basis[5] / N)} of the gradient signal and the abdomen {pct(basis[0] / N)}.</p>
        </div>
      </div>
    </Figure>
  );
}

/** Patient-level split, near-duplicate audit and scanner distribution per split — all from dataset_stats.json. */
export function SplitAudit() {
  const { datasetStats: ds, files } = useArtifacts();
  if (!ds) {
    return (
      <Figure title="Split audit: patients, near-duplicates, scanners" prov={files.datasetStats?.state === 'error' ? 'error' : 'missing'}
        caption="Patient counts per split, near-duplicate pairs across splits, and the ultrasound machine mix of each split, computed by notebook 01 and the bundle finaliser.">
        {files.datasetStats?.state === 'error' ? <FileError message={files.datasetStats.message} /> : <Empty title="No dataset statistics exported">Notebook 01 writes <code>dataset_stats.json</code>; the finaliser adds validation, near-duplicate and per-split scanner counts.</Empty>}
      </Figure>
    );
  }
  const sp = ds.split;
  const nd = ds.near_duplicates;
  const mbs = ds.machines_by_split;
  const splits = (['train', 'val', 'test'] as const).filter((k) => mbs?.[k]);
  const machines = mbs ? Array.from(new Set(splits.flatMap((k) => Object.keys(mbs[k] ?? {})))).sort() : [];
  // the machine whose share of the split changes most between training and test (computed, not assumed)
  const shift = mbs?.train && mbs?.test ? machines.map((m) => {
    const tr = mbs.train?.[m] ?? 0, te = mbs.test?.[m] ?? 0, va = mbs.val?.[m] ?? 0;
    const trN = sum(Object.values(mbs.train!)), teN = sum(Object.values(mbs.test!));
    return { m, tr, te, va, diff: te / teN - tr / trN };
  }).sort((a, b) => b.diff - a.diff)[0] : undefined;
  return (
    <Figure title="Split audit: patients, near-duplicates, scanners" prov="data" provDetail={`dataset_stats.json · ${ds.provenance.script}`}
      caption={<>All counts come from the metadata CSV, the saved split manifest and the near-duplicate scan of notebook 01. No model is involved. {readableProtocol(sp.protocol)}</>}>
      <div className="grid-3" style={{ alignItems: 'start' }}>
        <div className="stack">
          <b className="small">Patients and images per split</b>
          <table className="data small">
            <thead><tr><th></th><th className="num">Patients</th><th className="num">Images</th></tr></thead>
            <tbody>
              <tr><td>Train</td><td className="num">{fmtInt(sp.train_patients)}</td><td className="num">{fmtInt(sp.train_images)}</td></tr>
              {sp.val_patients != null && <tr><td>Validation</td><td className="num">{fmtInt(sp.val_patients)}</td><td className="num">{sp.val_images != null ? fmtInt(sp.val_images) : '—'}</td></tr>}
              <tr><td>Test</td><td className="num">{fmtInt(sp.test_patients)}</td><td className="num">{fmtInt(sp.test_images)}</td></tr>
            </tbody>
          </table>
          {sp.val_patients == null && <span className="small muted">Validation counts not in this bundle (schema 1.0): train includes validation.</span>}
          <p className="small" style={{ margin: 0 }}>Patients shared between splits: <b>{sp.patient_overlap}</b>. Every patient’s images sit in exactly one split, so the test set measures generalisation to new patients.</p>
        </div>
        <div className="stack">
          <b className="small">Near-duplicate images</b>
          {nd ? (
            <>
              <table className="data small">
                <tbody>
                  <tr><td>Detector</td><td>{nd.method}, Hamming ≤ {nd.threshold}</td></tr>
                  {nd.pairs != null && <tr><td>Similar pairs</td><td className="num">{fmtInt(nd.pairs)}</td></tr>}
                  <tr><td>Groups / images in groups</td><td className="num">{fmtInt(nd.groups)} / {fmtInt(nd.images_in_groups)}</td></tr>
                  {/* cross_split_pairs counts only (train or val) ↔ test pairs; train ↔ val pairs are listed separately */}
                  <tr><td>Pairs reaching the test set</td><td className="num"><b>{fmtInt(nd.cross_split_pairs)}</b></td></tr>
                  {nd.cross_split_by_pair && <>
                    <tr><td>&nbsp;· train–test</td><td className="num">{nd.cross_split_by_pair.train_test != null ? fmtInt(nd.cross_split_by_pair.train_test) : '—'}</td></tr>
                    <tr><td>&nbsp;· val–test</td><td className="num">{nd.cross_split_by_pair.val_test != null ? fmtInt(nd.cross_split_by_pair.val_test) : '—'}</td></tr>
                  </>}
                  {nd.test_images_involved != null && <tr><td>Test images involved</td><td className="num">{fmtInt(nd.test_images_involved)}</td></tr>}
                  {nd.cross_split_by_pair?.train_val != null && <tr><td>Train–validation pairs</td><td className="num">{fmtInt(nd.cross_split_by_pair.train_val)}</td></tr>}
                  {nd.label_discordant_fraction_cross_split != null && <tr><td>Cross-split pairs with different labels{nd.cross_split_by_pair ? ' (all splits)' : ''}</td><td className="num">{pct(nd.label_discordant_fraction_cross_split, 0)}</td></tr>}
                </tbody>
              </table>
              <p className="small muted" style={{ margin: 0 }}>
                These pairs belong to <i>different</i> patients (the split is patient-disjoint). A perceptual hash at this threshold is a loose detector: it also matches images that merely share layout, depth and overlays{nd.label_discordant_fraction_cross_split != null ? <>, which is why {pct(nd.label_discordant_fraction_cross_split, 0)} of cross-split pairs have different labels</> : ''}. Treat the count as an upper bound on visual redundancy, not as proven leakage.
              </p>
            </>
          ) : <Empty title="Near-duplicate statistics not in this bundle">Notebook 01 computes them (<code>splits/near_duplicates.csv</code>); the bundle finaliser copies them into <code>dataset_stats.json</code>.</Empty>}
        </div>
        <div className="stack">
          <b className="small">Ultrasound machine by split</b>
          {mbs && splits.length ? (
            <>
              <table className="data small">
                <thead><tr><th>Machine</th>{splits.map((k) => <th key={k} className="num">{k}</th>)}</tr></thead>
                <tbody>{machines.map((m) => <tr key={m} style={shift?.m === m ? { background: '#fbf3e4' } : undefined}><td>{m}</td>{splits.map((k) => <td key={k} className="num">{fmtInt(mbs[k]?.[m] ?? 0)}</td>)}</tr>)}</tbody>
              </table>
              {shift && shift.diff > 0 && <p className="small" style={{ margin: 0 }}><b>{shift.m}</b>: {fmtInt(shift.tr)} training, {mbs.val ? `${fmtInt(shift.va)} validation and ` : ''}{fmtInt(shift.te)} test images: its share of the test set is {(shift.diff * 100).toFixed(0)} percentage points higher than in training.{mbs.val && shift.va === 0 ? ' Validation contains none of its images, so model selection cannot reflect this scanner.' : ''} Chapter 4 reports accuracy per machine.</p>}
            </>
          ) : ds.machines ? (
            <>
              <table className="data small">
                <thead><tr><th>Machine</th><th className="num">Images (all splits)</th></tr></thead>
                <tbody>{Object.entries(ds.machines).map(([m, v]) => <tr key={m}><td>{m}</td><td className="num">{fmtInt(sum(v))}</td></tr>)}</tbody>
              </table>
              <span className="small muted">Per-split machine counts are not in this bundle (schema 1.0).</span>
            </>
          ) : <Empty title="Machine metadata not exported">—</Empty>}
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
    <Figure title="Image-level versus patient-level splitting" prov={datasetStats ? ['data', 'live', 'illustrative'] : ['live', 'illustrative']}
      caption={<>
        <b>Left</b>: if images are assigned to train/test independently with test fraction <i>f</i>, a test image from a patient with <i>n</i> images has a sibling in training with probability 1 − f<sup>n−1</sup>. This is exact. It uses {datasetStats ? <>the real images-per-patient distribution from the metadata CSV (mean {mean.toFixed(1)})</> : <>an <b>illustrative</b> distribution with the published mean of {mean.toFixed(1)} images per patient (replace by exporting dataset statistics)</>}.{' '}
        <b>Right (illustrative, synthetic data)</b>: a 1-nearest-neighbour “model” on synthetic features. Each patient contributes three acquisitions, each saved as several near-identical frames (FETAL_PLANES_DB file names such as <code>Patient00001_Plane1_3_of_15</code> indicate saved sequences). The gap between the two accuracies is the leakage. These percentages are <b>not</b> results of the workshop models.
      </>}>
      <div className="grid-2">
        <div className="stack">
          <Slider label="Test fraction f" value={testFrac} min={0.1} max={0.5} step={0.05} onChange={setTestFrac} format={(v) => pct(v, 0)} />
          <div style={{ fontFamily: 'var(--serif)', fontSize: '2.6rem', lineHeight: 1 }}>{pct(expected, 1)}</div>
          <div className="small">of test images would come from a patient who also has images in the training set under a random image-level split. With a patient-level split the figure is 0% by construction.</div>
          <p className="small muted" style={{ margin: 0 }}>The authors’ split (first 896 patients chronologically for training, next 896 for testing) is patient-disjoint and also temporal. Our validation set is carved patient-disjointly from the training half.</p>
        </div>
        <div className="stack">
          <span className="prov illustrative" style={{ alignSelf: 'start' }}>Illustrative · synthetic toy</span>
          <Slider label="Near-duplicate frames per acquisition" value={frames} min={1} max={10} step={1} onChange={setFrames} />
          <Slider label="Between-acquisition variability" value={spread} min={0.2} max={3} step={0.1} onChange={setSpread} format={(v) => v.toFixed(1)} />
          <HBars items={[
            { label: 'Random image split', value: sim.randomSplitAcc, color: 'var(--p-illus)' },
            { label: 'Patient-level split', value: sim.patientSplitAcc, color: 'var(--p-live)' },
          ]} max={1} format={(v) => `${pct(v)} (toy)`} />
          <div className="row small"><span>{sim.nImages} synthetic images · {pct(sim.leakedShare, 0)} of random-split test images have a same-patient sibling in train</span>
            <button className="btn" onClick={() => setSeed((s) => s + 1)}>Resample</button></div>
        </div>
      </div>
    </Figure>
  );
}
