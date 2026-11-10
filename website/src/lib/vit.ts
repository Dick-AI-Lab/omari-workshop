/**
 * ViT helpers: patch grids, the MAE fixed 2D sin-cos positional embedding (identical to
 * facebookresearch/mae util/pos_embed.py, which USF-MAE uses unchanged), random masking with the
 * same "argsort of uniform noise" scheme MAE uses, and a *toy* attention used only for illustration.
 */
import { mulberry32 } from './rng';
import { Gray, stats } from './imageOps';

/** 1D sin-cos embedding: ω_i = 1 / 10000^{i/(D/2)}, emb = [sin(p·ω), cos(p·ω)] */
export function sincos1d(dim: number, positions: number[]): number[][] {
  const half = dim / 2;
  const omega = Array.from({ length: half }, (_, i) => 1 / Math.pow(10000, i / half));
  return positions.map((p) => [...omega.map((w) => Math.sin(p * w)), ...omega.map((w) => Math.cos(p * w))]);
}

/**
 * MAE get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True).
 * meshgrid(grid_w, grid_h) → first half of the channels encodes the column (w) index, second half the row (h) index.
 * Returns (1 + G²) × D with an all-zero row for [CLS].
 */
export function sincos2d(dim: number, grid: number, clsToken = true): number[][] {
  const rows: number[][] = [];
  if (clsToken) rows.push(new Array(dim).fill(0));
  const half = dim / 2;
  for (let h = 0; h < grid; h++) {
    for (let w = 0; w < grid; w++) {
      const a = sincos1d(half, [w])[0];
      const b = sincos1d(half, [h])[0];
      rows.push([...a, ...b]);
    }
  }
  return rows;
}

export function cosine(a: number[], b: number[]): number {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return aa && bb ? ab / Math.sqrt(aa * bb) : 0;
}

/**
 * MAE random_masking: noise ~ U(0,1) per patch, keep the len_keep patches with the smallest noise.
 * Returns a boolean mask (true = masked / removed) over the G² patches.
 */
export function randomMask(numPatches: number, ratio: number, seed: number): boolean[] {
  const rand = mulberry32(seed);
  const noise = Array.from({ length: numPatches }, () => rand());
  const order = noise.map((v, i) => [v, i] as [number, number]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  const keep = Math.round(numPatches * (1 - ratio));
  const mask = new Array(numPatches).fill(true);
  for (let i = 0; i < keep; i++) mask[order[i]] = false;
  return mask;
}

/** Hand-crafted descriptor per 16×16 patch: [mean, std, |∂x|, |∂y|] standardised across patches. */
export function patchDescriptors(g: Gray, p: number): number[][] {
  const G = Math.floor(g.w / p);
  const raw: number[][] = [];
  for (let py = 0; py < G; py++) for (let px = 0; px < G; px++) {
    const vals: number[] = [];
    let gx = 0, gy = 0;
    for (let y = 0; y < p; y++) for (let x = 0; x < p; x++) {
      const X = px * p + x, Y = py * p + y;
      const v = g.data[Y * g.w + X];
      vals.push(v);
      if (x + 1 < p) gx += Math.abs(g.data[Y * g.w + X + 1] - v);
      if (y + 1 < p) gy += Math.abs(g.data[(Y + 1) * g.w + X] - v);
    }
    const s = stats(vals);
    raw.push([s.mean, s.std, gx / (p * p), gy / (p * p)]);
  }
  const d = raw[0].length;
  for (let j = 0; j < d; j++) {
    const col = raw.map((r) => r[j]);
    const s = stats(col);
    raw.forEach((r) => (r[j] = (r[j] - s.mean) / (s.std || 1)));
  }
  return raw;
}

/**
 * TOY self-attention for illustration only. Queries/keys are the hand-crafted descriptors concatenated with a
 * low-dimensional 2D sin-cos position code scaled by `posWeight`; no learned weights are involved.
 * A_ij = softmax_j( q_i · k_j / (√d · T) ).
 */
export function toyAttentionRow(desc: number[][], grid: number, query: number, posWeight: number, temperature: number): number[] {
  const pos = sincos2d(8, grid, false);
  const feats = desc.map((d, i) => [...d, ...pos[i].map((v) => v * posWeight)]);
  const q = feats[query];
  const dk = q.length;
  const logits = feats.map((k) => k.reduce((a, v, i) => a + v * q[i], 0) / (Math.sqrt(dk) * temperature));
  const m = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Entropy (nats) of an attention distribution, and its maximum log(n), for "how focused is this head?" */
export const entropy = (p: number[]) => -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0);
