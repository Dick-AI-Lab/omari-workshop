/** Small, exact summary statistics used to describe exported results (no modelling). */

export const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

export function mean(xs: (number | null | undefined)[]): number | null {
  const v = xs.filter(finite);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

/** Sample standard deviation (ddof = 1). Null when fewer than two finite values. */
export function sd(xs: (number | null | undefined)[]): number | null {
  const v = xs.filter(finite);
  if (v.length < 2) return null;
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
}

/** Holm–Bonferroni step-down adjusted p-values, in the input order. */
export function holm(ps: number[]): number[] {
  const m = ps.length;
  const order = ps.map((p, i) => [p, i] as [number, number]).sort((a, b) => a[0] - b[0]);
  const adj = new Array<number>(m);
  let running = 0;
  order.forEach(([p, i], k) => {
    running = Math.max(running, Math.min(1, (m - k) * p));
    adj[i] = running;
  });
  return adj;
}

/** Parse "Holm-adjusted p = 0.37" style notes written by the exporter. */
export function holmFromNote(note?: string | null): number | null {
  if (!note) return null;
  const m = /holm[^=]*=\s*([0-9.eE-]+)/i.exec(note);
  return m ? Number(m[1]) : null;
}
