/**
 * Toy models used to make mechanisms visible. Every output of this file must be labelled
 * "illustrative" or "live simulation" in the UI — none of it is a trained ultrasound model.
 */
import { mulberry32, gaussian, shuffleInPlace } from './rng';
import { Gray, conv2d, makeGray, relu, resize, annotationMask } from './imageOps';

/* ------------------------------------------------------------------ */
/* 1. Leakage simulator: near-duplicate frames + patient-level splitting */
/* ------------------------------------------------------------------ */

export interface LeakConfig {
  patients: number;
  acqPerPatient: number;      // acquisitions (one plane each) per patient
  framesPerAcq: number;       // near-identical frames saved per acquisition (cf. "PlaneX_k_of_n" naming)
  classSignal: number;        // separation between class centres
  acquisitionSpread: number;  // how different two acquisitions of the same class look
  frameNoise: number;         // frame-to-frame jitter within one acquisition
  testFraction: number;
  seed: number;
}

export interface LeakResult {
  randomSplitAcc: number;
  patientSplitAcc: number;
  leakedShare: number; // share of random-split test images whose patient also appears in train
  nImages: number;
}

const CLASSES = 6;
const DIM = 8;

/** 1-nearest-neighbour accuracy on synthetic features under image-level vs patient-level splits. */
export function simulateLeakage(cfg: LeakConfig): LeakResult {
  const rand = mulberry32(cfg.seed);
  const centres = Array.from({ length: CLASSES }, () => Array.from({ length: DIM }, () => gaussian(rand) * cfg.classSignal));
  const X: number[][] = [], y: number[] = [], pid: number[] = [];
  for (let p = 0; p < cfg.patients; p++) {
    for (let a = 0; a < cfg.acqPerPatient; a++) {
      const c = Math.floor(rand() * CLASSES);
      const acq = centres[c].map((v) => v + gaussian(rand) * cfg.acquisitionSpread);
      for (let f = 0; f < cfg.framesPerAcq; f++) {
        X.push(acq.map((v) => v + gaussian(rand) * cfg.frameNoise));
        y.push(c);
        pid.push(p);
      }
    }
  }
  const n = X.length;
  const nn = (trainIdx: number[], testIdx: number[]) => {
    let correct = 0;
    for (const i of testIdx) {
      let best = Infinity, lab = -1;
      for (const j of trainIdx) {
        let d = 0;
        for (let k = 0; k < DIM; k++) { const t = X[i][k] - X[j][k]; d += t * t; }
        if (d < best) { best = d; lab = y[j]; }
      }
      if (lab === y[i]) correct++;
    }
    return correct / Math.max(1, testIdx.length);
  };
  // image-level random split
  const all = shuffleInPlace(Array.from({ length: n }, (_, i) => i), rand);
  const nTest = Math.round(n * cfg.testFraction);
  const testR = all.slice(0, nTest), trainR = all.slice(nTest);
  const trainPatients = new Set(trainR.map((i) => pid[i]));
  const leaked = testR.filter((i) => trainPatients.has(pid[i])).length / Math.max(1, testR.length);
  // patient-level split
  const pts = shuffleInPlace(Array.from({ length: cfg.patients }, (_, i) => i), rand);
  const testP = new Set(pts.slice(0, Math.round(cfg.patients * cfg.testFraction)));
  const testG: number[] = [], trainG: number[] = [];
  for (let i = 0; i < n; i++) (testP.has(pid[i]) ? testG : trainG).push(i);
  return { randomSplitAcc: nn(trainR, testR), patientSplitAcc: nn(trainG, testG), leakedShare: leaked, nImages: n };
}

/**
 * Exact expectation for an image-level split with test fraction f: a test image from a patient with n images
 * is "leaked" unless all n−1 other images also fall in test → P(leak) = 1 − f^{n−1} (independent assignment).
 */
export function expectedLeakShare(imagesPerPatient: number[], f: number): number {
  let num = 0, den = 0;
  for (const n of imagesPerPatient) { num += n * (1 - Math.pow(f, n - 1)); den += n; }
  return den ? num / den : 0;
}

/* ------------------------------------------------------------------ */
/* 2. Toy CNN for CAM / Grad-CAM / Grad-CAM++ mechanics                 */
/* ------------------------------------------------------------------ */

export interface ToyChannel { id: string; label: string; }
export const TOY_CHANNELS: ToyChannel[] = [
  { id: 'bright', label: 'bright blobs' },
  { id: 'dark', label: 'dark regions' },
  { id: 'hedge', label: 'horizontal edges' },
  { id: 'vedge', label: 'vertical edges' },
  { id: 'texture', label: 'fine texture' },
  { id: 'overlay', label: 'burned-in overlay' },
];

const G = (s: number, r: number) => {
  const k: number[][] = [];
  for (let y = -r; y <= r; y++) { const row: number[] = []; for (let x = -r; x <= r; x++) row.push(Math.exp(-(x * x + y * y) / (2 * s * s))); k.push(row); }
  const sum = k.flat().reduce((a, b) => a + b, 0);
  return k.map((row) => row.map((v) => v / sum));
};

/** Feature maps A_k (K × 14 × 14), each ≥ 0, from fixed hand-crafted filters + ReLU + average pooling. */
export function toyFeatureMaps(img: Gray, grid = 14): Gray[] {
  const x = resize(img, 112, 112, 'area');
  const fov = makeGray(x.w, x.h);
  for (let i = 0; i < x.data.length; i++) fov.data[i] = x.data[i] > 0.03 ? 1 : 0; // imaging sector only
  const fine = conv2d(x, { kernel: G(1.5, 4), stride: 1, padding: 4, dilation: 1 });
  const coarse = conv2d(x, { kernel: G(6, 15), stride: 1, padding: 15, dilation: 1 });
  const dog = makeGray(x.w, x.h);
  for (let i = 0; i < dog.data.length; i++) dog.data[i] = fov.data[i] ? fine.data[i] - coarse.data[i] : 0; // difference of Gaussians
  const neg = makeGray(x.w, x.h); for (let i = 0; i < dog.data.length; i++) neg.data[i] = -dog.data[i];
  const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
  const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
  const lap = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]];
  const absIn = (g: Gray) => { const o = makeGray(g.w, g.h); for (let i = 0; i < g.data.length; i++) o.data[i] = fov.data[i] ? Math.abs(g.data[i]) : 0; return o; };
  const he = absIn(conv2d(fine, { kernel: sobelY, stride: 1, padding: 1, dilation: 1 }));
  const ve = absIn(conv2d(fine, { kernel: sobelX, stride: 1, padding: 1, dilation: 1 }));
  const tex = absIn(conv2d(x, { kernel: lap, stride: 1, padding: 1, dilation: 1 }));
  // overlays are thin, so detect them at full resolution and pool the coverage
  const ann = annotationMask(img, 0.93, 1);
  const annG = makeGray(img.w, img.h); for (let i = 0; i < ann.length; i++) annG.data[i] = ann[i];
  const maps = [relu(dog), relu(neg), he, ve, tex].map((m) => resize(m, grid, grid, 'area'));
  maps.push(resize(annG, grid, grid, 'area'));
  // scale each channel to unit max so weights are comparable
  return maps.map((m) => { let mx = 0; for (const v of m.data) mx = Math.max(mx, v); const o = makeGray(m.w, m.h); for (let i = 0; i < m.data.length; i++) o.data[i] = mx ? m.data[i] / mx : 0; return o; });
}

/** y = Σ_k w_k · GAP(A_k) + b */
export function toyScore(maps: Gray[], w: number[], b = 0): number {
  return maps.reduce((acc, m, k) => acc + w[k] * (m.data.reduce((a, v) => a + v, 0) / m.data.length), b);
}

export type ToyMethod = 'cam' | 'gradcam' | 'gradcampp';

/**
 * Attribution maps for the toy linear-GAP head (all analytical):
 *  CAM:        M = Σ_k w_k A_k
 *  Grad-CAM:   α_k = (1/Z) Σ_ij ∂y/∂A_kij = w_k / Z;  M = ReLU(Σ_k α_k A_k)  (∝ ReLU(CAM) for this head)
 *  Grad-CAM++: with S = exp(y), α_kij = g² / (2g² + Σ_ab A_kab g³), g = ∂y/∂A_kij = w_k / Z;
 *              w_k^{++} = Σ_ij α_kij · ReLU(g);  M = ReLU(Σ_k w_k^{++} A_k)
 */
export function toyAttribution(maps: Gray[], w: number[], method: ToyMethod): { map: Gray; channelWeights: number[] } {
  const g = maps[0].w;
  const Z = g * g;
  let cw: number[];
  if (method === 'cam') cw = w.slice();
  else if (method === 'gradcam') cw = w.map((v) => v / Z);
  else {
    cw = maps.map((m, k) => {
      const grad = w[k] / Z;
      if (grad === 0) return 0;
      const sumA = m.data.reduce((a, v) => a + v, 0);
      const alpha = (grad * grad) / (2 * grad * grad + sumA * grad * grad * grad);
      const safeAlpha = Number.isFinite(alpha) && alpha > 0 ? alpha : 0;
      return Z * safeAlpha * Math.max(0, grad);
    });
  }
  const out = makeGray(g, g);
  for (let i = 0; i < Z; i++) {
    let s = 0;
    for (let k = 0; k < maps.length; k++) s += cw[k] * maps[k].data[i];
    out.data[i] = method === 'cam' ? s : Math.max(0, s);
  }
  return { map: out, channelWeights: cw };
}

/**
 * Deletion curve on the toy model: progressively zero the most-attributed grid cells (in all channels)
 * and record the score. A faithful map should make the score fall quickly.
 */
export function toyDeletionCurve(maps: Gray[], w: number[], attribution: Gray, steps = 14): number[] {
  const order = Array.from(attribution.data).map((v, i) => [v, i] as [number, number]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
  const Z = attribution.data.length;
  const working = maps.map((m) => ({ w: m.w, h: m.h, data: new Float32Array(m.data) }));
  const out = [toyScore(working, w)];
  for (let s = 1; s <= steps; s++) {
    const upto = Math.round((s / steps) * Z);
    const from = Math.round(((s - 1) / steps) * Z);
    for (let r = from; r < upto; r++) working.forEach((m) => (m.data[order[r]] = 0));
    out.push(toyScore(working, w));
  }
  return out;
}

/** Normalise a map to [0, 1] for display (signed maps: negative values clipped to 0 for display only). */
export function unitScale(m: Gray, signed = false): Gray {
  let mx = -Infinity, mn = Infinity;
  for (const v of m.data) { mx = Math.max(mx, v); mn = Math.min(mn, v); }
  const o = makeGray(m.w, m.h);
  const lo = signed ? 0 : Math.min(0, mn);
  for (let i = 0; i < m.data.length; i++) o.data[i] = mx - lo > 0 ? Math.max(0, (m.data[i] - lo) / (mx - lo)) : 0;
  return o;
}
