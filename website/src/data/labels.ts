/**
 * Display conventions shared by every chapter (spec decisions 3 and 4).
 *  - Method id 'chefer' (schema 1.0) is the same computation as 'grad_rollout' (1.1): a simplified
 *    gradient-weighted attention rollout inspired by Chefer et al. (2021), without LRP.
 *  - ViT layer ids 'blockN' mean "input to model.blocks[N] (0-indexed)", i.e. the residual stream
 *    after N of 12 transformer blocks. They are displayed 1-indexed: blockN → "after block N of 12".
 *  - Attention JSON index i (0..11) is displayed as "Block i+1".
 */
import type { ModelId } from './types';

export const LEGACY_METHOD_IDS: Record<string, string> = { chefer: 'grad_rollout', chefer_lrp: 'grad_rollout' };
export const canonicalMethod = (id: string) => LEGACY_METHOD_IDS[id] ?? id;

export const METHOD_SHORT: Record<string, string> = {
  cam: 'CAM',
  gradcam: 'Grad-CAM',
  gradcampp: 'Grad-CAM++',
  vit_gradcam: 'Grad-CAM (tokens)',
  vit_gradcampp: 'Grad-CAM++ (tokens)',
  attention_rollout: 'Attention rollout',
  grad_rollout: 'Gradient-weighted attention rollout',
  metacampp: 'MetaCAM++',
  random: 'Random map (baseline)',
};

export const METHOD_LONG: Record<string, string> = {
  grad_rollout: 'Gradient-weighted attention rollout (simplified, inspired by Chefer et al. 2021; no LRP)',
};

export const methodShort = (id: string) => METHOD_SHORT[canonicalMethod(id)] ?? id;
export const methodLong = (id: string) => METHOD_LONG[canonicalMethod(id)] ?? methodShort(id);

/** Human-readable layer label; prefers an exporter-supplied label. */
export function layerLabel(model: ModelId | string, layer: string | null | undefined, supplied?: string | null): string {
  if (supplied) return supplied;
  if (!layer) return '—';
  const b = /^block(\d+)$/.exec(layer);
  if (b) return `after block ${Number(b[1])} of 12`;
  if (layer === 'final') return 'all 12 blocks (rollout)';
  if (layer === 'act') return model === 'densenet161' ? 'final features (norm5 + ReLU, 7×7)' : 'final features';
  const d = /denseblock(\d)$/.exec(layer);
  if (d) return `dense block ${d[1]} output`;
  return layer.replace('features.', '');
}

/** Short layer label for compact controls. */
export function layerShort(layer: string, supplied?: string | null): string {
  if (supplied) return supplied.replace(/ of 12$/, '');
  const b = /^block(\d+)$/.exec(layer);
  if (b) return `after block ${Number(b[1])}`;
  if (layer === 'final') return 'all blocks';
  if (layer === 'act') return 'final (7×7)';
  const d = /denseblock(\d)$/.exec(layer);
  if (d) return `dense block ${d[1]}`;
  return layer.replace('features.', '');
}

/** Rank used to order layers from shallow to deep. */
export function layerDepth(layer: string): number {
  const b = /^block(\d+)$/.exec(layer);
  if (b) return Number(b[1]);
  const d = /denseblock(\d)$/.exec(layer);
  if (d) return Number(d[1]) * 3;
  if (layer === 'act') return 13;
  if (layer === 'final') return 14;
  return 0;
}

export const shortSha = (s?: string | null) => (s ? s.replace(/^sha256:/, '').slice(0, 12) : undefined);

/** Detect which USF-MAE pretraining checkpoint a provenance string refers to. */
export function usfEpochs(checkpoint?: string | null): 100 | 500 | null {
  if (!checkpoint) return null;
  if (/500\s*ep|500ep/i.test(checkpoint)) return 500;
  if (/100\s*ep|100ep/i.test(checkpoint)) return 100;
  return null;
}
