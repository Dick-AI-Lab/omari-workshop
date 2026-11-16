import { ChapterHead } from '../components/shell';
import { Ask, Code, Deep, Tex } from '../components/ui';
import { CalibrationLab, DomainShiftTable, EvalOverlapNote, EvaluationSummary, LearningCurves, PrevalenceSimulator, SeedSpread, ThresholdLab } from '../visualizations/EvalLab';

export default function Ch4Evaluation() {
  return (
    <section className="chapter" id="evaluation">
      <ChapterHead id="evaluation" />
      <p className="lede">Pretrained weights are a starting point. Fine-tuning turns them into a plane classifier, and evaluation decides whether we believe the result.</p>

      <h3>Adapting a pretrained encoder</h3>
      <p>
        Every model gets a new linear head, <Tex>{'\\mathbb{R}^{D}\\to\\mathbb{R}^{6}'}</Tex>. In <b>linear probing</b> the encoder is frozen, so the result measures how linearly separable the planes already are in the pretrained representation. This is the cleanest comparison of what pretraining learned. In <b>full fine-tuning</b> every weight moves; it usually performs better but mixes the effect of pretraining with that of optimisation. Linear probes were run once per model at 100% labels; the label-efficiency sweep below uses full fine-tuning. Both use identical patient-disjoint splits, the same class-weighted loss and model selection by validation macro-F1. Learning rates, schedules and input transforms follow each model’s established recipe rather than a single setting that would favour one of them.
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
      <EvalOverlapNote />
      <SeedSpread />
      <p>
        Accuracy is dominated by the large classes. <b>Macro-F1</b> averages per-class F1, so the 711-image abdomen class counts as much as the 4,213-image “Other”. Look at the off-diagonal cells: confusions between abdomen and thorax, both axial trunk sections, are anatomically understandable. Confusions involving “Other” often reflect label ambiguity as much as model error.
      </p>
      <ThresholdLab />
      <PrevalenceSimulator />
      <CalibrationLab />
      <Deep kind="math">
        <Tex display>{'\\begin{gathered} \\mathrm{ECE} = \\sum_{b=1}^{B}\\frac{|S_b|}{n}\\,\\Big|\\,\\mathrm{acc}(S_b) - \\mathrm{conf}(S_b)\\Big| \\\\ \\hat p_i(T) = \\mathrm{softmax}(z_i / T),\\qquad T^\\star = \\arg\\min_T \\mathrm{NLL}_\\text{val}(T) \\end{gathered}'}</Tex>
      </Deep>

      <h3>When labels are scarce, and when the scanner changes</h3>
      <LearningCurves />
      <DomainShiftTable />
      <Ask q="Compare the label-efficiency curves with the full-label results above. What can we conclude?"
        options={['Self-supervision is useless with enough data', 'Domain pretraining may mainly buy label efficiency', 'The ViT architecture is better', 'Nothing firm without seeds, CIs and the overlap caveat']} multi
        reveal={<>Read the chart, not the headline. At 100% labels the three models are statistically indistinguishable on this test set. Where USF-MAE sits above DenseNet-161 at small label fractions, and the seed spreads do not overlap, the defensible reading is that the ultrasound-pretrained representation <i>may</i> reduce the labels needed, a valuable property for new planes or rare anomalies. Three qualifiers must travel with it: the models differ in architecture, pretraining data and fine-tuning recipe, not in one factor; without ViT-B/16, the architecture-matched control, at the same low label fractions (the chart shows whether it was run), the effect of ultrasound pretraining cannot be separated from the architecture; and USF-MAE’s pretraining corpus included FETAL_PLANES_DB images. Treat the result as suggestive and hypothesis-generating until it is confirmed on an external dataset.</>} />
    </section>
  );
}
