import { useState } from 'react';
import { ChapterHead } from '../components/shell';
import { Ask, ProvLegend } from '../components/ui';
import { useArtifacts } from '../data/artifacts';
import { CITATIONS } from '../data/literature';
import { OpeningChallenge } from './Ch1Opening';
import { CamToy } from '../visualizations/XaiLabs';
import { DetectiveLab, SanityCheckToy } from '../visualizations/Detective';

export function Break() {
  return (
    <section className="chapter" id="break">
      <ChapterHead id="break" />
      <p className="lede">Ten minutes. Everything on this page keeps working while you are away from the presenter’s screen.</p>
      <ul>
        <li>In the <a href="#data">preprocessing lab</a>, find the smallest added speckle that makes the histogram of the network input visibly change.</li>
        <li>In the <a href="#models">attention explorer</a>, set the position weight to zero. What happens to locality?</li>
        <li>In the <a href="#explain">toy CAM</a>, find weights for which Grad-CAM and Grad-CAM++ disagree most.</li>
      </ul>
    </section>
  );
}

export function Ch6Detective() {
  return (
    <section className="chapter" id="detective">
      <ChapterHead id="detective" />
      <p className="lede">A heatmap that lights up the skull feels reassuring. Reassurance is not evidence. The detective’s rule: state a hypothesis, then run the experiment that could prove it wrong.</p>
      <p>
        Shortcut learning means solving the benchmark with a cue that is predictive in the data but irrelevant to the task (Geirhos et al., 2020). In obstetric ultrasound, candidate shortcuts are everywhere: on-screen text that names a preset, caliper marks placed only on biometry planes, depth and zoom settings that differ by plane, the sector shape of a particular probe, or a machine used mainly for cervical scans. Each is testable by changing that one factor and nothing else.
      </p>
      <DetectiveLab />
      <p>
        <b>Plausible</b> means a map agrees with what an expert expects to see. <b>Faithful</b> means it reflects what the model actually computes. The two can come apart in both directions. A map can look anatomical while the decision rests on an overlay, and a model can use the right anatomy while the method fails to show it at 7×7 resolution. Faithfulness is assessed with interventions: occlude what the map highlights, randomise the weights, perturb the input, and see whether the model and the map change as they should.
      </p>
      <CamToy shortcut />
      <SanityCheckToy />
      <Ask q="Occluding the region Grad-CAM highlights drops the true-class probability from 0.97 to 0.40. What has this shown?"
        options={['The model uses the correct anatomy', 'The highlighted region matters to this model', 'Grad-CAM is faithful in general', 'Occlusion itself may be out of distribution']} multi
        reveal={<>The second and fourth. A large drop shows the region is influential for this image and this model. It does not show the region is <i>anatomically correct</i>, and it does not validate the method on other images. A grey or black occluder is also an unusual input, so part of the drop can come from distribution shift rather than missing evidence. Better practice compares with occluding a random region of the same size, uses in-distribution infills, and reports deletion and insertion curves over many images.</>} />
    </section>
  );
}

const CHECKLIST = [
  'External validation on another hospital, machine vendor and population',
  'Patient-level splits and a check for near-duplicates across splits',
  'Per-class and subgroup results with uncertainty (machine, operator, gestational age, BMI)',
  'Calibration on the deployment population, with a fitted temperature or equivalent',
  'Shortcut tests: overlays removed, presets varied, probe geometry varied',
  'Faithfulness checks for any explanation shown to users (randomisation, deletion)',
  'Prospective, workflow-integrated evaluation with sonographers in the loop',
  'Failure-mode review: what happens on unseen planes, anomalies, poor-quality frames',
  'Pretraining-data audit: overlap between foundation-model corpus and evaluation data',
  'Regulatory and governance pathway: intended use, monitoring, change control',
];

export function Ch7Closing() {
  const { manifest, status, backend } = useArtifacts();
  const [done, setDone] = useState<number[]>([]);
  return (
    <section className="chapter" id="closing">
      <ChapterHead id="closing" />
      <p className="lede">Back to the first image. You now know what each model was pretrained on, how it was fine-tuned, how it scores, and what its explanation can and cannot tell you.</p>
      <OpeningChallenge />
      <Ask q="Which model would you investigate further, and why?" options={['DenseNet-161', 'ViT-B/16', 'USF-MAE', 'None yet']}
        reveal={<>There is no single right answer; the reasoning is what matters. A good answer names the evidence that would change your mind: label-efficiency curves with seeds, external validation, calibration, and shortcut tests. It also states the caveat about USF-MAE’s pretraining overlap with this dataset.</>} />
      <h3>What would it take to deploy?</h3>
      <p>Tick what has been shown in this workshop. Most of the list cannot be shown on one public dataset, and that is the point.</p>
      <div className="ask">
        {CHECKLIST.map((c, i) => (
          <label key={c} className="row small" style={{ flexWrap: 'nowrap', alignItems: 'flex-start', margin: '0.3rem 0' }}>
            <input type="checkbox" checked={done.includes(i)} onChange={(e) => setDone(e.target.checked ? [...done, i] : done.filter((x) => x !== i))} /> {c}
          </label>
        ))}
        <p className="small muted" style={{ marginBottom: 0 }}>{done.length} of {CHECKLIST.length} ticked.</p>
      </div>
      <h3>What to take away</h3>
      <ul>
        <li>Prediction quality, confidence and reasoning are separate properties, measured by separate tools.</li>
        <li>Preprocessing and splitting are modelling decisions. Leakage can make any architecture look excellent.</li>
        <li>CNNs build in locality; ViTs learn relationships globally; self-supervised domain pretraining changes <i>what</i> the encoder already knows. Comparisons are between recipes, not single factors.</li>
        <li>Macro-averaged metrics, calibration and label-efficiency curves say more than accuracy alone.</li>
        <li>CAM-family maps answer a narrow question at coarse resolution. Attention is routing, not attribution. Every explanation needs a faithfulness test.</li>
      </ul>
      <h3>Run the experiments yourself</h3>
      <p>The six notebooks are standalone and run in Google Colab. Each has a quick demo mode (subset, few epochs) and a full mode.</p>
      <ol className="small" style={{ fontFamily: 'var(--sans)' }}>
        <li><code>notebooks/01_dataset_preparation.ipynb</code>: metadata, splits, duplicates, statistics for this page</li>
        <li><code>notebooks/02_densenet161.ipynb</code>: frozen and fine-tuned DenseNet-161, feature maps</li>
        <li><code>notebooks/03_vit_b16.ipynb</code>: ViT-B/16, patch embeddings, attention export</li>
        <li><code>notebooks/04_usf_mae.ipynb</code>: USF-MAE checkpoint verification, fine-tuning, reconstructions</li>
        <li><code>notebooks/05_model_evaluation.ipynb</code>: metrics, CIs, calibration, label efficiency, by-machine results</li>
        <li><code>notebooks/06_explainability.ipynb</code>: CAM, Grad-CAM, Grad-CAM++, ViT attribution, faithfulness tests</li>
      </ol>
      <h3>How to read the figures on this page</h3>
      <ProvLegend />
      <p className="small muted" style={{ fontFamily: 'var(--sans)' }}>
        Artifact bundle: {status === 'ready' ? `${manifest!.bundle_id} (schema ${manifest!.schema_version}, generated ${manifest!.generated_at}, ${manifest!.samples.length} samples, models: ${manifest!.models.map((m) => m.display_name).join(', ')})` : status === 'absent' ? 'none loaded. All model-output figures show their empty state.' : status}.
        {' '}Live inference server: {backend.ok ? `connected at ${backend.url}` : 'not running'}.
      </p>
      <h3>References</h3>
      <ol className="small" style={{ fontFamily: 'var(--sans)', lineHeight: 1.5 }}>
        {CITATIONS.map((c) => <li key={c.key}>{c.text} <a href={c.url}>{c.url.replace('https://', '')}</a></li>)}
      </ol>
    </section>
  );
}
