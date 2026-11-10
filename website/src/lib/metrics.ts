/** Classification metrics, calibration and prior-shift arithmetic. All exact; no model involved. */

export const softmax = (z: number[], T = 1): number[] => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp((v - m) / T));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

/** cm[i][j] = count(true = i, pred = j) */
export function confusionMatrix(labels: number[], preds: number[], k: number): number[][] {
  const cm = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < labels.length; i++) cm[labels[i]][preds[i]]++;
  return cm;
}

export interface PerClass { precision: number; recall: number; f1: number; support: number; }

export function perClass(cm: number[][]): PerClass[] {
  const k = cm.length;
  return Array.from({ length: k }, (_, c) => {
    const tp = cm[c][c];
    const support = cm[c].reduce((a, b) => a + b, 0);
    const predicted = cm.reduce((a, row) => a + row[c], 0);
    const precision = predicted ? tp / predicted : 0;
    const recall = support ? tp / support : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1, support };
  });
}

export function summary(cm: number[][]) {
  const pc = perClass(cm);
  const total = cm.flat().reduce((a, b) => a + b, 0);
  const correct = cm.reduce((a, row, i) => a + row[i], 0);
  return {
    accuracy: total ? correct / total : 0,
    macroF1: pc.reduce((a, p) => a + p.f1, 0) / pc.length,
    balancedAccuracy: pc.reduce((a, p) => a + p.recall, 0) / pc.length,
    perClass: pc,
    total,
  };
}

/**
 * Prior (label) shift: keep the classifier's class-conditional behaviour P(ŷ | y) fixed,
 * change the deployment prevalences π_y. Expected confusion: E[cm'] = N · π_y · P(ŷ | y).
 * Precision changes with prevalence; recall does not. This assumes P(x | y) is unchanged.
 */
export function shiftPrevalence(cm: number[][], priors: number[], n = 10000): number[][] {
  const k = cm.length;
  const s = priors.reduce((a, b) => a + b, 0) || 1;
  return Array.from({ length: k }, (_, i) => {
    const row = cm[i];
    const rs = row.reduce((a, b) => a + b, 0) || 1;
    return row.map((v) => (n * (priors[i] / s) * v) / rs);
  });
}

export interface ReliabilityBin { lo: number; hi: number; conf: number; acc: number; count: number; }

/** Top-label reliability diagram and expected calibration error with equal-width bins. */
export function reliability(probs: number[][], labels: number[], bins = 10): { bins: ReliabilityBin[]; ece: number; mce: number } {
  const out: ReliabilityBin[] = Array.from({ length: bins }, (_, b) => ({ lo: b / bins, hi: (b + 1) / bins, conf: 0, acc: 0, count: 0 }));
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    let arg = 0;
    for (let c = 1; c < p.length; c++) if (p[c] > p[arg]) arg = c;
    const conf = p[arg];
    const b = Math.min(bins - 1, Math.floor(conf * bins));
    out[b].conf += conf;
    out[b].acc += arg === labels[i] ? 1 : 0;
    out[b].count++;
  }
  let ece = 0, mce = 0;
  for (const b of out) {
    if (b.count) { b.conf /= b.count; b.acc /= b.count; }
    const gap = Math.abs(b.acc - b.conf);
    ece += (b.count / Math.max(1, probs.length)) * gap;
    if (b.count) mce = Math.max(mce, gap);
  }
  return { bins: out, ece, mce };
}

export function nll(probs: number[][], labels: number[]): number {
  let s = 0;
  for (let i = 0; i < probs.length; i++) s -= Math.log(Math.max(1e-12, probs[i][labels[i]]));
  return s / Math.max(1, probs.length);
}

/** One-vs-rest ROC curve and AUC (trapezoidal) for class c. */
export function rocCurve(scores: number[], positives: boolean[]): { fpr: number[]; tpr: number[]; thr: number[]; auc: number } {
  const idx = scores.map((s, i) => [s, i] as [number, number]).sort((a, b) => b[0] - a[0]);
  const P = positives.filter(Boolean).length, N = positives.length - P;
  const fpr = [0], tpr = [0], thr = [Infinity];
  let tp = 0, fp = 0;
  for (let k = 0; k < idx.length; k++) {
    const [s, i] = idx[k];
    if (positives[i]) tp++; else fp++;
    if (k === idx.length - 1 || idx[k + 1][0] !== s) { fpr.push(N ? fp / N : 0); tpr.push(P ? tp / P : 0); thr.push(s); }
  }
  let auc = 0;
  for (let k = 1; k < fpr.length; k++) auc += (fpr[k] - fpr[k - 1]) * (tpr[k] + tpr[k - 1]) / 2;
  return { fpr, tpr, thr, auc };
}

/** Inverse-frequency class weights w_c = N / (K · n_c), the 'balanced' heuristic used by scikit-learn. */
export function balancedWeights(counts: number[]): number[] {
  const N = counts.reduce((a, b) => a + b, 0);
  return counts.map((n) => (n ? N / (counts.length * n) : 0));
}

/** Effective-number weights (Cui et al., CVPR 2019): w_c ∝ (1 − β) / (1 − β^{n_c}), normalised to sum K. */
export function effectiveNumberWeights(counts: number[], beta: number): number[] {
  const raw = counts.map((n) => (1 - beta) / (1 - Math.pow(beta, n)));
  const s = raw.reduce((a, b) => a + b, 0);
  return raw.map((w) => (w * counts.length) / s);
}
