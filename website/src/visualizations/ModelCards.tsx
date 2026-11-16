import { OverlapNote } from '../components/Disclosure';
import { Figure, MODEL_META, TechNotes, fmtM } from '../components/ui';
import { useArtifacts } from '../data/artifacts';
import { shortSha } from '../data/labels';
import type { ModelId } from '../data/types';
import { densenet161Stages, totalParams, vitParams } from '../lib/architectures';

/** Configuration defaults (configs/*.yaml and spec decision 1); replaced by bundle values where present. */
const DEFAULTS: Record<ModelId, { arch: string; init: string; data: string; objective: string; input: string; bias: string }> = {
  densenet161: {
    arch: 'DenseNet-161 CNN (4 dense blocks, GAP + linear head)',
    init: 'torchvision DenseNet161_Weights.IMAGENET1K_V1',
    data: 'ImageNet-1k (1.28 M labelled natural images)',
    objective: 'Supervised 1000-way classification',
    input: '224², ImageNet μ/σ, bilinear',
    bias: 'Locality, translation equivariance, hierarchy, feature reuse',
  },
  vit_b16: {
    arch: 'ViT-B/16 (12 blocks, 12 heads, 768-d, [CLS] head)',
    init: 'timm vit_base_patch16_224.augreg_in1k',
    data: 'ImageNet-1k (AugReg recipe, heavy augmentation)',
    objective: 'Supervised 1000-way classification',
    input: '224², μ = σ = 0.5, bicubic',
    bias: 'Patchification only; global attention from layer 1; learned positions',
  },
  usf_mae: {
    arch: 'ViT-B/16 encoder (identical to the column on the left)',
    init: 'USF-MAE 500-epoch pretrained checkpoint usf_mae_500ep.pth (encoder), itself initialised from Facebook MAE ViT-B/16 (ImageNet-1k)',
    data: 'OpenUS-46: ≈370 k unlabelled ultrasound images, 46 datasets, including FETAL_PLANES_DB',
    objective: 'Masked autoencoding (75% masking, pixel MSE)',
    input: '224² stretched, ImageNet μ/σ, nearest',
    bias: 'Same as ViT-B/16, plus ultrasound statistics learned without labels; fixed sin-cos positions',
  },
};

export function ModelComparisonTable() {
  const { manifest } = useArtifacts();
  const params: Record<ModelId, number> = { densenet161: totalParams(densenet161Stages(6)), vit_b16: vitParams(undefined, 6).total, usf_mae: vitParams(undefined, 6).total };
  const ids: ModelId[] = ['densenet161', 'vit_b16', 'usf_mae'];
  const info = (id: ModelId) => manifest?.models.find((m) => m.id === id);
  const pre = (id: ModelId) => info(id)?.pretraining;
  return (
    <Figure title="Same image, three ways of having learned to see" prov={manifest ? ['live', 'model'] : 'live'}
      caption={<>Parameter counts are computed from the architecture definitions (6-class heads). Pretraining and fine-tuning rows are the defaults in <code>configs/*.yaml</code>{manifest ? '; where the artifact bundle records a value, it replaces the default' : ''}. Read the table in three layers: <b>architecture</b> (ViT-B/16 and USF-MAE are identical networks), <b>pretraining data and objective</b> (ImageNet labels versus unlabelled ultrasound), and <b>fine-tuning recipe</b> (learning rate, schedule, layer decay, input transform, all following each model’s established recipe). The models differ in all three, so the comparison is between <b>recipes</b>, not a controlled ablation of one factor.</>}>
      <div style={{ overflowX: 'auto' }}>
        <table className="data">
          <thead><tr><th></th>{ids.map((id) => <th key={id} style={{ color: MODEL_META[id].text }}>{MODEL_META[id].name}</th>)}</tr></thead>
          <tbody>
            <tr><th colSpan={4} className="small muted">Architecture</th></tr>
            <tr><th>Network</th>{ids.map((id) => <td key={id}>{DEFAULTS[id].arch}</td>)}</tr>
            <tr><th>Parameters</th>{ids.map((id) => <td key={id}>{fmtM(params[id])}</td>)}</tr>
            <tr><th>Inductive bias</th>{ids.map((id) => <td key={id}>{DEFAULTS[id].bias}</td>)}</tr>
            <tr><th colSpan={4} className="small muted">Pretraining (before any fetal label)</th></tr>
            <tr><th>Initialisation</th>{ids.map((id) => <td key={id}>{info(id)?.initialization ?? DEFAULTS[id].init}</td>)}</tr>
            <tr><th>Pretraining checkpoint</th>{ids.map((id) => { const p = pre(id); return <td key={id}>{p?.checkpoint ? <>{p.checkpoint}{p.epochs ? ` · ${p.epochs} epochs` : ''}{p.sha256 ? <span className="muted small"> · sha256 {shortSha(p.sha256)}…</span> : ''}</> : id === 'usf_mae' ? 'usf_mae_500ep.pth (500 epochs)' : '—'}</td>; })}</tr>
            <tr><th>Pretraining data</th>{ids.map((id) => <td key={id}>{pre(id)?.corpus ?? DEFAULTS[id].data}</td>)}</tr>
            <tr><th>Pretraining objective</th>{ids.map((id) => <td key={id}>{pre(id)?.objective ?? DEFAULTS[id].objective}</td>)}</tr>
            <tr><th colSpan={4} className="small muted">Fine-tuning recipe (this workshop)</th></tr>
            <tr><th>Input transform</th>{ids.map((id) => { const m = info(id); return <td key={id}>{m ? `${m.input.size}², ${m.input.resize}, ${m.input.interpolation}, μ=${m.input.mean.map((v) => v.toFixed(3)).join('/')}` : DEFAULTS[id].input}</td>; })}</tr>
            {manifest && <tr><th>Fine-tuning</th>{ids.map((id) => <td key={id}>{info(id)?.finetuning ?? '—'}</td>)}</tr>}
            {manifest && <tr><th>Classifier checkpoint</th>{ids.map((id) => <td key={id} className="small">{info(id)?.checkpoint_id ?? '—'}{info(id)?.checkpoint_sha256 ? <span className="muted"> · {shortSha(info(id)!.checkpoint_sha256)}…</span> : ''}{info(id)?.checkpoint_status?.matches_summary === false && <div style={{ color: 'var(--p-missing)' }}>File differs from its run summary (interrupted re-run); see the provenance notes in Chapter 7.</div>}</td>)}</tr>}
            {manifest?.models.some((m) => m.notes) && <tr><th>Notes</th>{ids.map((id) => <td key={id} className="small">{info(id)?.notes ? <TechNotes notes={[info(id)!.notes]} summary="Details" /> : '—'}</td>)}</tr>}
          </tbody>
        </table>
      </div>
      <OverlapNote context="USF-MAE" />
    </Figure>
  );
}
