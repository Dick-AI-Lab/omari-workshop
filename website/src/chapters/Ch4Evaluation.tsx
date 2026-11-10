import { ChapterHead } from '../components/shell';
import { Ask, Code, Deep, Tex } from '../components/ui';
import { CalibrationLab, DomainShiftTable, EvaluationSummary, LearningCurves, PrevalenceSimulator, ThresholdLab } from '../visualizations/EvalLab';

export default function Ch4Evaluation() {
  return (
    <section className="chapter" id="evaluation">
      <ChapterHead id="evaluation" />
      <p className="lede">Pretrained weights are a starting point. Fine-tuning turns them into a plane classifier, and evaluation decides whether we believe the result.</p>

      <h3>Adapting a pretrained encoder</h3>
      <p>
        Every model gets a new linear head, <Tex>{'\\mathbb{R}^{D}\\to\\mathbb{R}^{6}'}</Tex>. In <b>linear probing</b> the encoder is frozen, so the result measures how linearly separable the planes already are in the pretrained representation. This is the cleanest comparison of what pretraining learned. In <b>full fine-tuning</b> every weight moves; it usually performs better but mixes the effect of pretraining with that of optimisation. We run both, with identical patient-disjoint splits, the same class-weighted loss and model selection by validation macro-F1. Learning rates, schedules and input transforms follow each model’s established recipe rather than a single setting that would favour one of them.
      </p>
      <Deep kind="math">
        <p>Class-weighted cross-entropy with label smoothing <Tex>\varepsilon</Tex>, and the focal-loss alternative:</p>
        <Tex display>{'\\mathcal{L}_\\text{wCE} = -\\frac{1}{\\sum_i w_{y_i}}\\sum_i w_{y_i} \\sum_c \\tilde y_{ic}\\log p_{ic},\\quad \\tilde y_{ic} = (1-\\varepsilon)\\,[c = y_i] + \\tfrac{\\varepsilon}{K}'}</Tex>
        <Tex display>{'\\mathcal{L}_\\text{focal} = -\\sum_i \\alpha_{y_i}(1 - p_{i y_i})^{\\gamma}\\log p_{i y_i}'}</Tex>
        <p>Layer-wise learning-rate decay for transformers (MAE / BEiT recipe): block <Tex>\ell</Tex> of <Tex>L</Tex> gets <Tex>{'\\eta_\\ell = \\eta\\,\\lambda^{L+1-\\ell}'}</Tex>, so early, more generic layers move least.</p>
      </Deep>
      <Deep kind="code">
        <Code>{`from omari_us.training import TrainConfig, train

cfg = TrainConfig(model="usf_mae", regime="finetune", epochs=30, batch_size=64,
                  lr=5e-4, layer_decay=0.65, weight_decay=0.05, warmup_epochs=3,
                  loss="weighted_ce", label_smoothing=0.1, label_fraction=1.0, seed=0)
result = train(cfg, splits="artifacts/splits/split_manifest.csv", data_root=DATA_ROOT)
# saves best.pt (by val macro-F1), history.json and test_predictions.npz with full provenance`}</Code>
      </Deep>

      <h3>What the test set says</h3>
      <EvaluationSummary />
      <p>
        Accuracy is dominated by the large classes. <b>Macro-F1</b> averages per-class F1, so the 711-image abdomen class counts as much as the 4,213-image “Other”. Look at the off-diagonal cells: confusions between abdomen and thorax, both axial trunk sections, are anatomically understandable. Confusions involving “Other” often reflect label ambiguity as much as model error.
      </p>
      <ThresholdLab />
      <PrevalenceSimulator />
      <CalibrationLab />
      <Deep kind="math">
        <Tex display>{'\\mathrm{ECE} = \\sum_{b=1}^{B}\\frac{|S_b|}{n}\\,\\Big|\\,\\mathrm{acc}(S_b) - \\mathrm{conf}(S_b)\\Big|,\\qquad \\hat p_i(T) = \\mathrm{softmax}(z_i / T),\\ \\ T^\\star = \\arg\\min_T \\mathrm{NLL}_\\text{val}(T)'}</Tex>
      </Deep>

      <h3>When labels are scarce, and when the scanner changes</h3>
      <LearningCurves />
      <DomainShiftTable />
      <Ask q="USF-MAE is best at 5% labels but ties DenseNet at 100%. What can we conclude?"
        options={['Self-supervision is useless with enough data', 'Domain pretraining mainly buys label efficiency', 'The ViT architecture is better', 'Nothing without CIs and the overlap caveat']} multi
        reveal={<>If (and only if) the experiment shows that pattern with non-overlapping uncertainty across seeds, the defensible reading is that the ultrasound-pretrained representation mainly reduces the labels needed. That is a valuable property for new planes or rare anomalies. Two qualifiers must travel with it. The models differ in more than pretraining data (architecture, objective, augmentation), and USF-MAE’s pretraining corpus included FETAL_PLANES_DB images. Treat the result as hypothesis-generating until it is confirmed on an external dataset.</>} />
    </section>
  );
}
