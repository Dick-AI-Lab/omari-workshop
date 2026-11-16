import { ReactNode } from 'react';
import { useArtifacts } from '../data/artifacts';
import { USF_MAE_LITERATURE as USF } from '../data/literature';
import { shortSha } from '../data/labels';
import { TechNotes } from './ui';

/**
 * USF-MAE pretraining-overlap disclosure. Reads the manifest (models[].notes / pretraining, schema 1.1);
 * falls back to the project's literature note when a 1.0 bundle dropped it.
 */
export function OverlapNote({ context, children }: { context?: string; children?: ReactNode }) {
  const { manifest } = useArtifacts();
  const usf = manifest?.models.find((m) => m.id === 'usf_mae');
  const raw = (usf?.pretraining?.overlap_with_test || usf?.notes)?.trim();
  // bundle text may end without a full stop; the fixed sentence below follows it directly
  const fromBundle = raw && !/[.!?]$/.test(raw) ? `${raw}.` : raw;
  return (
    <div className="disclosure" role="note">
      <b>Pretraining overlap{context ? ` (${context})` : ''}.</b>{' '}
      {fromBundle ? fromBundle : <>{USF.overlap} <span className="muted">(project note; this bundle carries no model note)</span></>}{' '}
      Unlabelled exposure is not supervised label leakage, but it is a transductive advantage that ImageNet-initialised models do not have, and it may favour USF-MAE most when labels are scarce.
      {children}
    </div>
  );
}

/** One-line description of the USF-MAE pretraining checkpoint, from the bundle. */
export function usfCheckpointText(manifest?: ReturnType<typeof useArtifacts>['manifest']): string {
  const usf = manifest?.models.find((m) => m.id === 'usf_mae');
  const p = usf?.pretraining;
  if (p?.checkpoint) return `${p.checkpoint}${p.epochs ? ` (${p.epochs}-epoch pretraining)` : ''}${p.sha256 ? `, sha256 ${shortSha(p.sha256)}…` : ''}`;
  if (usf?.initialization) return usf.initialization;
  return 'USF-MAE 500-epoch pretrained checkpoint (usf_mae_500ep.pth)';
}

/** Global provenance caveats shipped with a 1.1 bundle. */
export function ProvenanceNotes() {
  const { manifest } = useArtifacts();
  const src = manifest?.sample_predictions_source;
  // the finaliser also repeats sample_predictions_source inside provenance_notes; list it once
  const notes = (manifest?.provenance_notes ?? []).filter((n) => !src || n.slice(0, 60) !== src.slice(0, 60));
  if (!notes.length && !src) return null;
  return <TechNotes summary="Provenance notes for this artifact bundle" notes={[src ? `Per-sample predictions: ${src}` : null, ...notes]} />;
}

export function NotRun({ children }: { children: ReactNode }) {
  return <span className="notrun">{children}</span>;
}
