/**
 * Exact architecture arithmetic (no weights needed): layer shapes, parameter counts and theoretical
 * receptive fields for DenseNet-161 (Huang et al., 2017; torchvision definition) and ViT-B/16
 * (Dosovitskiy et al., 2021; timm definition), plus the MAE decoder used by USF-MAE
 * (mae_vit_base_patch16_dec512d8b).
 */

export interface RFStage {
  name: string;
  kind: 'conv' | 'pool' | 'block' | 'transition' | 'head';
  /** output spatial size for a 224×224 input */
  size: number;
  channels: number;
  /** theoretical receptive field (pixels) of one output unit, along the longest path */
  rf: number;
  /** cumulative stride ("jump") in input pixels */
  jump: number;
  params: number;
}

export const DENSENET161 = { initFeatures: 96, growth: 48, blocks: [6, 12, 36, 24], bnSize: 4 } as const;

const bn = (c: number) => 2 * c; // affine weight + bias (running stats are buffers, not parameters)

/** Parameter count of one dense layer: BN-ReLU-Conv1×1(→bn_size·k)-BN-ReLU-Conv3×3(→k), no conv bias. */
export function denseLayerParams(cin: number, growth: number = DENSENET161.growth, bnSize: number = DENSENET161.bnSize): number {
  const mid = bnSize * growth;
  return bn(cin) + cin * mid + bn(mid) + mid * growth * 9;
}

export function densenet161Stages(numClasses = 1000, input = 224): RFStage[] {
  const { initFeatures, growth, blocks } = DENSENET161;
  const stages: RFStage[] = [];
  let rf = 1, jump = 1, size = input, ch = 3;
  const conv = (k: number, s: number, p: number) => { rf += (k - 1) * jump; jump *= s; size = Math.floor((size + 2 * p - k) / s) + 1; };
  conv(7, 2, 3);
  stages.push({ name: 'conv0 7×7/2', kind: 'conv', size, channels: initFeatures, rf, jump, params: ch * initFeatures * 49 + bn(initFeatures) });
  ch = initFeatures;
  conv(3, 2, 1);
  stages.push({ name: 'pool0 max 3×3/2', kind: 'pool', size, channels: ch, rf, jump, params: 0 });
  blocks.forEach((n, bi) => {
    let p = 0;
    for (let l = 0; l < n; l++) { p += denseLayerParams(ch + l * growth); rf += 2 * jump; /* 3×3 conv, 1×1 adds nothing */ }
    ch += n * growth;
    stages.push({ name: `denseblock${bi + 1} (${n} layers)`, kind: 'block', size, channels: ch, rf, jump, params: p });
    if (bi < blocks.length - 1) {
      const out = Math.floor(ch / 2);
      const tp = bn(ch) + ch * out;
      conv(2, 2, 0); // avgpool 2×2/2 after 1×1 conv
      stages.push({ name: `transition${bi + 1}`, kind: 'transition', size, channels: out, rf, jump, params: tp });
      ch = out;
    }
  });
  stages.push({ name: `norm5 + GAP + linear(${ch}→${numClasses})`, kind: 'head', size: 1, channels: numClasses, rf, jump, params: bn(ch) + ch * numClasses + numClasses });
  return stages;
}

export const totalParams = (s: RFStage[]) => s.reduce((a, b) => a + b.params, 0);

/** Channel count entering each layer of a dense block: c_ℓ = c_0 + ℓ·k */
export const denseInputs = (c0: number, layers: number, k: number = DENSENET161.growth) => Array.from({ length: layers + 1 }, (_, l) => c0 + l * k);

export interface ViTSpec { patch: number; dim: number; depth: number; heads: number; mlpRatio: number; img: number; inCh: number; }
export const VIT_B16: ViTSpec = { patch: 16, dim: 768, depth: 12, heads: 12, mlpRatio: 4, img: 224, inCh: 3 };
export const MAE_DECODER = { dim: 512, depth: 8, heads: 16, mlpRatio: 4 };

export function vitBlockParams(d: number, mlpRatio = 4): number {
  const h = d * mlpRatio;
  return 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + 2 * d + (d * h + h) + (h * d + d);
}

export function vitParams(spec: ViTSpec = VIT_B16, numClasses = 1000) {
  const n = (spec.img / spec.patch) ** 2;
  const patchEmbed = spec.patch * spec.patch * spec.inCh * spec.dim + spec.dim;
  const tokens = spec.dim /* cls */ + (n + 1) * spec.dim; /* pos */
  const blocks = spec.depth * vitBlockParams(spec.dim, spec.mlpRatio);
  const norm = 2 * spec.dim;
  const head = numClasses ? spec.dim * numClasses + numClasses : 0;
  return { tokensN: n, patchEmbed, tokens, blocks, norm, head, total: patchEmbed + tokens + blocks + norm + head };
}

export function maeDecoderParams(enc = VIT_B16, dec = MAE_DECODER) {
  const n = (enc.img / enc.patch) ** 2;
  const embed = enc.dim * dec.dim + dec.dim;
  const maskToken = dec.dim;
  const pos = (n + 1) * dec.dim; // fixed sin-cos (requires_grad = False) but stored in the state dict
  const blocks = dec.depth * vitBlockParams(dec.dim, dec.mlpRatio);
  const norm = 2 * dec.dim;
  const pred = dec.dim * enc.patch * enc.patch * enc.inCh + enc.patch * enc.patch * enc.inCh;
  return { embed, maskToken, pos, blocks, norm, pred, total: embed + maskToken + pos + blocks + norm + pred };
}

/**
 * Approximate multiply–accumulates for one ViT encoder forward on `tokens` tokens:
 * per block 4·N·d² (QKV + proj) + 2·N²·d (QKᵀ and A·V) + 2·N·d·h (MLP).
 */
export function vitEncoderMACs(tokens: number, d = 768, depth = 12, mlpRatio = 4): number {
  return depth * (4 * tokens * d * d + 2 * tokens * tokens * d + 2 * tokens * d * d * mlpRatio);
}
