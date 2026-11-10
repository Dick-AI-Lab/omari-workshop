import { Figure, MODEL_META, fmtM } from '../components/ui';
import { useArtifacts } from '../data/artifacts';
import type { ModelId } from '../data/types';
import { densenet161Stages, totalParams, vitParams } from '../lib/architectures';

const DEFAULTS: Record<ModelId, { init: string; data: string; objective: string; input: string; bias: string }> = {
  densenet161: {
    init: 'torchvision DenseNet161_Weights.IMAGENET1K_V1',
    data: 'ImageNet-1k (1.28 M labelled natural images)',
    objective: 'Supervised 1000-way classification',
    input: '224², ImageNet μ/σ, bilinear',
    bias: 'Locality, translation equivariance, hierarchy, feature reuse',
  },
  vit_b16: {
    init: 'timm vit_base_patch16_224.augreg_in1k',
    data: 'ImageNet-1k (AugReg recipe, heavy augmentation)',
    objective: 'Supervised 1000-way classification',
    input: '224², μ = σ = 0.5, bicubic',
    bias: 'Patchification only; global attention from layer 1; learned positions',
  },
  usf_mae: {
    init: 'USF-MAE 100-epoch checkpoint (encoder), which was itself initialised from Facebook MAE ViT-B/16 (ImageNet-1k)',
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
  return (
    <Figure title="Same image, three ways of having learned to see" prov={manifest ? ['live', 'model'] : 'live'}
      caption={<>Parameter counts are computed from the architecture definitions (6-class heads). Initialisation and preprocessing are the defaults in <code>configs/*.yaml</code>{manifest ? '; where an artifact bundle is loaded, its recorded values replace these' : ''}. The three models differ in architecture <i>and</i> in pretraining data and objective, so the comparison is between <b>recipes</b>, not a controlled ablation of one factor.</>}>
      <div style={{ overflowX: 'auto' }}>
        <table className="data">
          <thead><tr><th></th>{ids.map((id) => <th key={id} style={{ color: MODEL_META[id].text }}>{MODEL_META[id].name}</th>)}</tr></thead>
          <tbody>
            <tr><th>Parameters</th>{ids.map((id) => <td key={id}>{fmtM(params[id])}</td>)}</tr>
            <tr><th>Initialisation</th>{ids.map((id) => <td key={id}>{manifest?.models.find((m) => m.id === id)?.initialization ?? DEFAULTS[id].init}</td>)}</tr>
            <tr><th>Pretraining data</th>{ids.map((id) => <td key={id}>{DEFAULTS[id].data}</td>)}</tr>
            <tr><th>Pretraining objective</th>{ids.map((id) => <td key={id}>{DEFAULTS[id].objective}</td>)}</tr>
            <tr><th>Input transform</th>{ids.map((id) => { const m = manifest?.models.find((x) => x.id === id); return <td key={id}>{m ? `${m.input.size}², ${m.input.resize}, ${m.input.interpolation}, μ=${m.input.mean.map((v) => v.toFixed(3)).join('/')}` : DEFAULTS[id].input}</td>; })}</tr>
            <tr><th>Inductive bias</th>{ids.map((id) => <td key={id}>{DEFAULTS[id].bias}</td>)}</tr>
            {manifest && <tr><th>Fine-tuning</th>{ids.map((id) => <td key={id}>{manifest.models.find((m) => m.id === id)?.finetuning ?? '—'}</td>)}</tr>}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}
