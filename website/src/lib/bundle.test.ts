import { describe, expect, it } from 'vitest';
import { holm, holmFromNote, mean, sd } from './stats';
import { canonicalMethod, layerLabel, methodLong, usfEpochs } from '../data/labels';
import { normalizeManifest } from '../data/artifacts';
import { summarizeGrid } from '../visualizations/LabelEfficiency';
import type { LabelEfficiency, Manifest } from '../data/types';

describe('stats', () => {
  it('ignores nulls (not run) instead of treating them as 0', () => {
    expect(mean([0.9, null, 0.8])).toBeCloseTo(0.85);
    expect(sd([0.9, null])).toBeNull();
    expect(sd([1, 2, 3])).toBeCloseTo(1); // ddof = 1
  });
  it('Holm adjustment matches the step-down definition', () => {
    const a = holm([0.01, 0.04, 0.03]);
    expect(a[0]).toBeCloseTo(0.03);
    expect(a[2]).toBeCloseTo(0.06);
    expect(a[1]).toBeCloseTo(0.06);
    expect(holm([0.7, 0.8, 1])).toEqual([1, 1, 1]);
    expect(holmFromNote('Holm-adjusted p = 1; image-level pairing')).toBe(1);
  });
});

describe('labels (spec decisions 3 and 4)', () => {
  it('maps the legacy chefer id and labels it honestly', () => {
    expect(canonicalMethod('chefer')).toBe('grad_rollout');
    expect(methodLong('chefer')).toMatch(/inspired by Chefer et al\. 2021; no LRP/);
  });
  it('labels ViT layers 1-indexed as the residual stream after N blocks', () => {
    expect(layerLabel('vit_b16', 'block11')).toBe('after block 11 of 12');
    expect(layerLabel('vit_b16', 'block5')).toBe('after block 5 of 12');
    expect(layerLabel('vit_b16', 'block5', 'supplied')).toBe('supplied');
  });
  it('detects USF-MAE checkpoint epochs', () => {
    expect(usfEpochs('usf_mae_500ep.pth')).toBe(500);
    expect(usfEpochs('usf_mae_100ep.pth')).toBe(100);
    expect(usfEpochs('usf_mae_finetune_lf1_s0/best.pt')).toBeNull();
  });
});

describe('schema 1.0 → 1.1 normalisation', () => {
  it('renames chefer attributions and fills display labels', () => {
    const m = normalizeManifest({
      schema_version: '1.0', models: [], classes: [],
      samples: [{ id: 's', attributions: [{ model: 'vit_b16', method: 'chefer', layer: 'final', target_class: 0 }], perturbations: [{ attributions: { vit_b16: { method: 'chefer', png: 'x' } } }] }],
    } as unknown as Manifest);
    const a = m.samples[0].attributions![0];
    expect(a.method).toBe('grad_rollout');
    expect(a.legacy_method).toBe('chefer');
    expect(a.layer_label).toBe('all 12 blocks (rollout)');
    expect(m.samples[0].perturbations![0].attributions!.vit_b16!.method).toBe('grad_rollout');
  });
});

describe('label-efficiency summary', () => {
  it('reports not-run cells as n = 0 and single seeds without sd', () => {
    const le = { provenance: {} as never, regime: 'finetune', fractions: [0.01, 1], seeds: [0, 1, 2],
      per_model: { vit_b16: { macro_f1: [[null, null, null], [0.94, null, null]] }, usf_mae: { macro_f1: [[0.6, 0.7, 0.65], [0.94, 0.94, 0.95]] } } } as LabelEfficiency;
    const v = summarizeGrid(le, 'vit_b16', 'macro_f1');
    expect(v[0].n).toBe(0);
    expect(v[0].mean).toBeNull();
    expect(v[1].n).toBe(1);
    expect(v[1].sd).toBeNull();
    expect(summarizeGrid(le, 'usf_mae', 'macro_f1')[0].sd).toBeCloseTo(0.05);
    expect(summarizeGrid(le, 'usf_mae', 'accuracy')).toEqual([]);
  });
});
