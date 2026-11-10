import { CLASS_SHORT } from '../data/literature';
import type { ModelId, Prediction } from '../data/types';
import { MODEL_META } from '../components/ui';

/** Three models' class-probability vectors, aligned by class so disagreement is visible at a glance. */
export function PredictionBars({ preds, truth, highlight }: { preds: Partial<Record<ModelId, Prediction>>; truth?: number; highlight?: number }) {
  const models = (Object.keys(MODEL_META) as ModelId[]).filter((m) => preds[m]);
  const W = 560, rowH = 30, lw = 84, colW = (W - lw) / Math.max(1, models.length);
  const H = CLASS_SHORT.length * rowH + 28;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Predicted class probabilities per model">
      {models.map((m, mi) => (
        <text key={m} x={lw + mi * colW + 4} y={14} style={{ fill: MODEL_META[m].text, fontWeight: 600 }}>{MODEL_META[m].name}</text>
      ))}
      {CLASS_SHORT.map((c, ci) => (
        <g key={c} transform={`translate(0, ${24 + ci * rowH})`}>
          <rect x={0} y={0} width={W} height={rowH - 4} fill={ci === truth ? '#e6f1ea' : ci === highlight ? '#eef2f4' : 'transparent'} rx={3} />
          <text x={lw - 8} y={rowH / 2 + 2} textAnchor="end" style={{ fill: 'var(--ink)', fontWeight: ci === truth ? 650 : 400 }}>{c}{ci === truth ? ' ✓' : ''}</text>
          {models.map((m, mi) => {
            const p = preds[m]!;
            const v = p.probs[ci] ?? 0;
            const isPred = p.pred === ci;
            const bw = (colW - 52) * v;
            return (
              <g key={m} transform={`translate(${lw + mi * colW + 4}, 4)`}>
                <rect x={0} y={0} width={colW - 52} height={rowH - 12} fill="#edf1f3" rx={2} />
                <rect x={0} y={0} width={Math.max(0.5, bw)} height={rowH - 12} fill={MODEL_META[m].color} opacity={isPred ? 1 : 0.55} rx={2} />
                <text x={colW - 48} y={rowH / 2 - 1} style={{ fontVariantNumeric: 'tabular-nums', fill: isPred ? 'var(--ink)' : 'var(--ink-3)', fontWeight: isPred ? 600 : 400 }}>{(v * 100).toFixed(v < 0.1 ? 1 : 0)}%</text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}
