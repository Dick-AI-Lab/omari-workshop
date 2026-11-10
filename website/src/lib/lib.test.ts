import { describe, expect, it } from 'vitest';
import { densenet161Stages, maeDecoderParams, totalParams, vitParams } from './architectures';
import { addSpeckle, conv2d, fitToSquare, makeGray, normalize, resize, stats } from './imageOps';
import { balancedWeights, confusionMatrix, perClass, reliability, rocCurve, shiftPrevalence, softmax, summary } from './metrics';
import { expectedLeakShare, toyAttribution } from './toyModels';
import { randomMask, sincos2d } from './vit';

describe('architecture arithmetic matches the reference implementations', () => {
  it('DenseNet-161 parameter counts equal torchvision (1000 and 6 classes)', () => {
    expect(totalParams(densenet161Stages(1000))).toBe(28_681_000);
    expect(totalParams(densenet161Stages(6))).toBe(26_485_254);
  });
  it('DenseNet-161 final width and output size', () => {
    const s = densenet161Stages(6);
    const block4 = s.find((x) => x.name.startsWith('denseblock4'))!;
    expect(block4.channels).toBe(2208);
    expect(block4.size).toBe(7);
  });
  it('ViT-B/16 parameter counts equal timm vit_base_patch16_224', () => {
    expect(vitParams(undefined, 1000).total).toBe(86_567_656);
    expect(vitParams(undefined, 6).total).toBe(85_803_270);
  });
  it('MAE decoder parameter count is positive and dominated by blocks', () => {
    const d = maeDecoderParams();
    expect(d.blocks).toBe(8 * 3_152_384);
    expect(d.pred).toBe(512 * 768 + 768);
  });
});

describe('MAE positional embedding', () => {
  it('matches facebookresearch/mae get_2d_sincos_pos_embed values', () => {
    const e = sincos2d(768, 14, true);
    expect(e.length).toBe(197);
    expect(e[0].every((v) => v === 0)).toBe(true);
    // reference values computed with the MAE numpy implementation for token 16 (row 1, col 1)
    const ref: [number, number][] = [[0, 0.841471], [1, 0.815251], [383, 1.0], [384, 0.841471], [385, 0.815251], [767, 1.0]];
    for (const [i, v] of ref) expect(e[16][i]).toBeCloseTo(v, 5);
  });
  it('random masking keeps round(N(1-r)) patches', () => {
    const m = randomMask(196, 0.75, 3);
    expect(m.filter((x) => !x).length).toBe(49);
  });
});

describe('image operations', () => {
  it('identity kernel convolution with padding 1 is the identity', () => {
    const g = makeGray(5, 4);
    g.data.forEach((_, i) => (g.data[i] = i / 20));
    const o = conv2d(g, { kernel: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], stride: 1, padding: 1, dilation: 1 });
    expect(Array.from(o.data)).toEqual(Array.from(g.data));
  });
  it('conv output size follows the standard formula', () => {
    const g = makeGray(32, 32, 1);
    const o = conv2d(g, { kernel: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], stride: 2, padding: 0, dilation: 2 });
    expect(o.w).toBe(Math.floor((32 - 5) / 2) + 1);
  });
  it('letterbox preserves aspect ratio and stretch fills the square', () => {
    const g = makeGray(400, 200, 0.5);
    const lb = fitToSquare(g, 224, 'letterbox');
    expect(lb.content.w).toBe(224);
    expect(lb.content.h).toBe(112);
    const st = fitToSquare(g, 224, 'stretch');
    expect(st.out.w).toBe(224);
    expect(st.scaleX).not.toBeCloseTo(st.scaleY);
  });
  it('ImageNet normalisation of mid-grey matches (0.5 - mu) / sigma', () => {
    const g = makeGray(4, 4, 0.5);
    const n = normalize(g, 'imagenet');
    expect(n.channels![0][0]).toBeCloseTo((0.5 - 0.485) / 0.229, 6);
    expect(n.channels![2][0]).toBeCloseTo((0.5 - 0.406) / 0.225, 6);
  });
  it('multiplicative speckle has mean ≈ 1 and variance ≈ 1/L at full strength', () => {
    const g = makeGray(200, 200, 0.4);
    const L = 4;
    const o = addSpeckle(g, 1, L, 9);
    const s = stats(Array.from(o.data).map((v) => v / 0.4));
    expect(s.mean).toBeCloseTo(1, 1);
    expect(s.std ** 2).toBeCloseTo(1 / L, 1);
  });
  it('area resize preserves the mean', () => {
    const g = makeGray(64, 64);
    g.data.forEach((_, i) => (g.data[i] = (i % 7) / 7));
    expect(stats(resize(g, 16, 16, 'area').data).mean).toBeCloseTo(stats(g.data).mean, 2);
  });
});

describe('metrics', () => {
  it('per-class metrics and macro-F1 on a known confusion matrix', () => {
    const cm = confusionMatrix([0, 0, 1, 1, 1, 2], [0, 1, 1, 1, 2, 2], 3);
    const pc = perClass(cm);
    expect(pc[0].recall).toBeCloseTo(0.5);
    expect(pc[1].precision).toBeCloseTo(2 / 3);
    const s = summary(cm);
    expect(s.accuracy).toBeCloseTo(4 / 6);
  });
  it('prior shift leaves recall unchanged and moves precision', () => {
    const cm = [[90, 10], [5, 95]];
    const a = summary(shiftPrevalence(cm, [0.5, 0.5]));
    const b = summary(shiftPrevalence(cm, [0.05, 0.95]));
    expect(a.perClass[0].recall).toBeCloseTo(b.perClass[0].recall, 9);
    expect(b.perClass[0].precision).toBeLessThan(a.perClass[0].precision);
  });
  it('a perfectly calibrated, perfectly separated set has zero ECE', () => {
    const probs = [[1, 0], [0, 1], [1, 0]];
    expect(reliability(probs, [0, 1, 0]).ece).toBeCloseTo(0);
  });
  it('ROC AUC of a perfect ranking is 1 and of a reversed ranking is 0', () => {
    expect(rocCurve([0.9, 0.8, 0.2, 0.1], [true, true, false, false]).auc).toBeCloseTo(1);
    expect(rocCurve([0.1, 0.2, 0.8, 0.9], [true, true, false, false]).auc).toBeCloseTo(0);
  });
  it('balanced weights equal N / (K n_c)', () => {
    expect(balancedWeights([10, 30])).toEqual([2, 2 / 3]);
  });
  it('softmax sums to one and temperature preserves argmax', () => {
    const p = softmax([1, 2, 3], 0.5);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p.indexOf(Math.max(...p))).toBe(2);
  });
});

describe('toy models', () => {
  it('leak expectation: singletons never leak, large patients almost always do', () => {
    expect(expectedLeakShare([1, 1, 1], 0.2)).toBe(0);
    expect(expectedLeakShare([50], 0.2)).toBeCloseTo(1, 6);
  });
  it('Grad-CAM equals ReLU(CAM)/Z for a linear GAP head', () => {
    const maps = [makeGray(2, 2), makeGray(2, 2)];
    maps[0].data.set([1, 0, 0, 1]); maps[1].data.set([0, 1, 1, 0]);
    const w = [2, -1];
    const cam = toyAttribution(maps, w, 'cam').map.data;
    const gc = toyAttribution(maps, w, 'gradcam').map.data;
    cam.forEach((v, i) => expect(gc[i]).toBeCloseTo(Math.max(0, v) / 4));
  });
  it('Grad-CAM++ channel weight matches the closed form relu(w)/(2 + w·mean(A))', () => {
    const m = makeGray(2, 2); m.data.set([1, 1, 0, 0]);
    const w = [3];
    const r = toyAttribution([m], w, 'gradcampp');
    const Z = 4, g = w[0] / Z, sumA = 2;
    const alpha = 1 / (2 + sumA * g);
    expect(r.channelWeights[0]).toBeCloseTo(Z * alpha * g, 9);
  });
});
