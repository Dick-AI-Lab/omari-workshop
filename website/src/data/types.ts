/**
 * Artifact bundle schema — the contract between the Python export pipeline and the website.
 * The site accepts schema 1.0 (python/omari_us/export, original Colab bundle) and schema 1.1
 * (scripts/finalize_website_bundle.py, which adds the optional fields marked "1.1" below).
 * Mirrored in docs/ARTIFACT_SCHEMA.md. All paths are relative to website/public/experiments/.
 * Every 1.1 field is optional so a 1.0 bundle still type-checks; absent data must be shown as
 * "not run / not exported", never substituted.
 */

export type ModelId = 'densenet161' | 'vit_b16' | 'usf_mae';
export const MODEL_IDS: ModelId[] = ['densenet161', 'vit_b16', 'usf_mae'];

export interface Provenance {
  script: string;
  generated_at: string;
  checkpoint_id?: string | null;
  checkpoint_sha256?: string | null;
  params?: Record<string, unknown>;
  git_commit?: string | null;
  note?: string | null;
  [k: string]: unknown;
}

/** 1.1: what the encoder was pretrained on, kept separate from fine-tuning. */
export interface Pretraining {
  checkpoint?: string | null;
  sha256?: string | null;
  epochs?: number | null;
  objective?: string | null;
  corpus?: string | null;
  overlap_with_test?: string | null;
}

export interface ModelInfo {
  id: ModelId;
  display_name: string;
  checkpoint_id: string;
  checkpoint_sha256?: string | null;
  initialization: string;
  finetuning: string;
  params: number;
  input: { size: number; resize: 'stretch' | 'letterbox' | 'centercrop' | string; interpolation: string; mean: number[]; std: number[]; channels: number };
  /** 1.0: often null. 1.1: non-null; for usf_mae it carries the pretraining-overlap disclosure. */
  notes?: string | null;
  pretraining?: Pretraining | null;
  /** 1.1, densenet161: the classifier file differs from its run summary (see provenance_notes). */
  checkpoint_status?: { best_pt_sha256?: string; summary_checkpoint_sha256?: string; matches_summary?: boolean; functional_equivalence?: string | null } | null;
}

export interface Prediction { probs: number[]; pred: number; }

export interface FeatureChannel { index: number; png: string; min: number; max: number; mean: number; }
export interface FeatureLayer { layer: string; shape: [number, number, number]; channels: FeatureChannel[]; selection: string; }

export interface AttentionEntry {
  /** JSON file: { blocks, heads, grid, cls: number[block][head][196] (raw softmax probabilities of the [CLS] query row, excluding [CLS] key), rollout?: number[] } */
  cls: string;
  /** optional full matrices for some blocks (0-indexed): little-endian uint16, shape heads × (N+1) × (N+1), raw probability × 65535 */
  full?: { block: number; path: string; heads: number; tokens: number }[];
}

export interface AttributionEntry {
  model: ModelId;
  /** 'cam' | 'gradcam' | 'gradcampp' | 'vit_gradcam' | 'vit_gradcampp' | 'attention_rollout' | 'grad_rollout' (1.0: 'chefer') | 'metacampp' */
  method: string;
  layer: string;
  target_class: number;
  png: string; // 8-bit grayscale heatmap, min–max scaled; raw range below
  raw_min: number;
  raw_max: number;
  resolution: [number, number];
  provenance: Provenance;
  /** 1.1 display strings (the loader fills them for 1.0 bundles). */
  method_label?: string;
  layer_label?: string;
  /** set by the loader when a 1.0 id was renamed (e.g. 'chefer' → 'grad_rollout'). */
  legacy_method?: string;
}

export interface ReconstructionEntry {
  mask_ratio: number;
  seed: number;
  masked: number[]; // masked patch indices (row-major over the 14×14 grid)
  reconstruction_png: string; // full-image reconstruction (decoder output, unpatchified, de-normalised)
  pasted_png?: string; // visible patches from input + reconstructed masked patches
  loss_masked_mse: number;
  provenance: Provenance;
}

export interface PerturbationEntry {
  type: 'contrast' | 'brightness' | 'speckle' | 'blur' | 'resolution' | 'occlude_annotations' | 'occlude_patch' | 'flip' | string;
  param: number;
  label: string;
  image_png: string;
  predictions: Partial<Record<ModelId, Prediction>>;
  attributions?: Partial<Record<ModelId, { method: string; png: string; layer?: string; layer_label?: string; method_label?: string; target_class?: number; legacy_method?: string }>>;
  provenance: Provenance;
}

export interface SampleInfo {
  id: string;
  image: string;
  width: number;
  height: number;
  label: number;
  brain_plane?: string;
  machine?: string;
  operator?: string;
  split: 'train' | 'val' | 'test';
  patient: string; // the dataset's own pseudonymous id, e.g. Patient00042
  tags?: string[]; // e.g. 'disagreement', 'misclassified-<model>', 'opening', 'class-k'
  /** 1.1: from the headline run's test_predictions.npz (original-resolution evaluation). */
  predictions: Partial<Record<ModelId, Prediction>>;
  /** 1.1: the 1.0 values recomputed on the exported (≤640 px) PNG — same pipeline as perturbations[].predictions. */
  predictions_display_png?: Partial<Record<ModelId, Prediction>>;
  features?: Partial<Record<ModelId, FeatureLayer[]>>;
  attention?: Partial<Record<ModelId, AttentionEntry>>;
  attributions?: AttributionEntry[];
  /** primary reconstructions (1.1: 500-epoch checkpoint when generated; check provenance.checkpoint_id) */
  reconstructions?: ReconstructionEntry[];
  /** 1.1: original reconstructions from the historical 100-epoch checkpoint, provenance unchanged */
  reconstructions_legacy?: ReconstructionEntry[];
  perturbations?: PerturbationEntry[];
}

export interface Manifest {
  schema_version: string;
  bundle_id: string;
  generated_at: string;
  generator: { script: string; version: string; git_commit?: string | null; command?: string | null };
  dataset: { name: string; license: string; source_url: string; citation: string; split_protocol: string };
  classes: string[];
  models: ModelInfo[];
  samples: SampleInfo[];
  dataset_stats?: string;
  evaluation?: string;
  training_curves?: string;
  label_efficiency?: string;
  /** 1.1 */
  faithfulness?: string;
  provenance_notes?: string[];
  sample_predictions_source?: string;
}

export interface PerModelEval {
  confusion: number[][];
  accuracy: number;
  macro_f1: number;
  balanced_accuracy: number;
  per_class: { precision: number; recall: number; f1: number; support: number }[];
  ci95?: { accuracy?: [number, number]; macro_f1?: [number, number]; balanced_accuracy?: [number, number] };
  ece?: number | null;
  nll?: number | null;
  by_machine?: Record<string, { n: number; accuracy: number; macro_f1: number }>;
  /** temperature fitted on the validation set (source field name: temperature_fit_on_val; 1.1 may also use `temperature`) */
  temperature_fit_on_val?: number | null;
  temperature?: number | null;
  ece_after_temperature?: number | null;
}

export interface SeedSummary {
  seeds: number[];
  runs?: string[];
  accuracy: (number | null)[];
  macro_f1: (number | null)[];
  mean?: { accuracy?: number | null; macro_f1?: number | null };
  sd?: { accuracy?: number | null; macro_f1?: number | null };
  sd_ddof?: number;
  sd_definition?: string;
  note?: string;
}

export interface Comparison {
  a: ModelId; b: ModelId; test: string; statistic: number; p_value: number; note?: string;
  p_holm?: number | null; p_value_holm?: number | null; p_adjusted?: number | null;
}

export interface Evaluation {
  provenance: Provenance;
  split: string;
  n: number;
  per_model: Partial<Record<ModelId, PerModelEval>>;
  test_predictions?: { ids: string[]; labels: number[]; machines?: string[]; probs: Partial<Record<ModelId, number[][]>> };
  misclassified?: { id: string; image: string; label: number; preds: Partial<Record<ModelId, number>> }[];
  comparisons?: Comparison[];
  /** 1.1: seed-to-seed spread of headline (100% labels) fine-tuning runs */
  seed_summary?: Partial<Record<ModelId, SeedSummary>>;
  calibration_fields?: Record<string, string>;
  notes?: string[];
}

export interface TrainingCurves {
  provenance: Provenance;
  per_model: Partial<Record<ModelId, Record<string, { epoch: number[]; train_loss: number[]; val_loss: number[]; val_macro_f1: number[]; lr?: number[] }>>>;
  notes?: string[];
}

/** fractions × seeds grids; null = run not available. */
export type Grid = (number | null)[][];

export interface LabelEfficiency {
  provenance: Provenance;
  fractions: number[];
  seeds: number[];
  regime: string;
  per_model: Partial<Record<ModelId, { macro_f1: Grid; accuracy?: Grid; runs?: (string | null)[][] }>>;
  /** 1.1 */
  availability?: Partial<Record<ModelId, string>>;
  notes?: string[];
}

export interface NearDuplicates {
  method: string;
  threshold: number;
  groups: number;
  images_in_groups: number;
  cross_split_pairs: number;
  /** 1.1 */
  pairs?: number;
  cross_split_by_pair?: { train_val?: number; train_test?: number; val_test?: number };
  test_images_involved?: number;
  label_discordant_fraction_cross_split?: number;
  note?: string;
}

export interface DatasetStats {
  provenance: Provenance;
  total_images: number;
  total_patients: number;
  classes: string[];
  /** 1.0: train includes validation. 1.1: train excludes val; train_plus_val = authors' training half. */
  class_counts: { train: number[]; test: number[]; val?: number[]; train_plus_val?: number[] };
  brain_planes?: Record<string, number>;
  machines?: Record<string, number[]>; // machine → per-class counts (all splits)
  operators?: Record<string, number>;
  images_per_patient: number[]; // one entry per patient
  split: {
    train_images: number; test_images: number; train_patients: number; test_patients: number; patient_overlap: number;
    val_images?: number; val_patients?: number; protocol?: string;
  };
  near_duplicates?: NearDuplicates;
  /** 1.1 */
  machines_by_split?: { train?: Record<string, number>; val?: Record<string, number>; test?: Record<string, number> };
}

/** 1.1 faithfulness.json (deletion/insertion AUC over a random subset of test images). */
export interface FaithfulnessRow {
  model: ModelId;
  method: string;
  layer?: string | null;
  deletion_auc_mean: number | null;
  deletion_auc_sd: number | null;
  insertion_auc_mean: number | null;
  insertion_auc_sd: number | null;
  method_label?: string;
  layer_label?: string;
}

export interface Faithfulness {
  provenance: Provenance;
  n_images?: number | null;
  steps?: number | null;
  baseline?: string | null;
  target?: string | null;
  rows: FaithfulnessRow[];
  random_baseline?: FaithfulnessRow[] | Partial<Record<ModelId, Partial<FaithfulnessRow>>> | null;
  notes?: string[] | string | null;
}
