import { useRef } from 'react';
import { useArtifacts } from '../data/artifacts';
import { useImageSource } from '../data/imageSource';
import { CLASS_SHORT } from '../data/literature';

/** Shared image selector. Changing the image here updates every linked lab on the page. */
export function ImagePicker({ compact = false }: { compact?: boolean }) {
  const src = useImageSource();
  const { manifest } = useArtifacts();
  const fileRef = useRef<HTMLInputElement>(null);
  const samples = manifest?.samples ?? [];
  return (
    <div className="row" style={{ gap: '0.5rem 0.8rem', marginBottom: compact ? '0.6rem' : '1rem' }}>
      <label className="control-label" htmlFor="img-src" style={{ gap: '0.4rem' }}><span>Image</span></label>
      <select id="img-src" value={src.kind === 'sample' ? src.sample?.id : src.kind}
        onChange={(e) => {
          const v = e.target.value;
          if (v === 'phantom') src.selectPhantom();
          else if (v === 'upload') fileRef.current?.click();
          else src.selectSample(v);
        }}>
        <option value="phantom">Synthetic phantom (not a patient image)</option>
        {samples.length > 0 && (
          <optgroup label="FETAL_PLANES_DB samples (exported)">
            {samples.map((s) => <option key={s.id} value={s.id}>{s.id} · {CLASS_SHORT[s.label] ?? s.label}{s.tags?.length ? ` · ${s.tags.join(', ')}` : ''}</option>)}
          </optgroup>
        )}
        <option value="upload">{src.kind === 'upload' ? `Your image: ${src.name}` : 'Load your own image…'}</option>
      </select>
      <input ref={fileRef} type="file" accept="image/png,image/jpeg" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void src.selectUpload(f); e.target.value = ''; }} />
      {src.loading && <span className="muted small">loading…</span>}
      {!compact && (
        <span className="muted small">
          {src.kind === 'phantom' && 'Geometric phantom with simulated speckle, generated in your browser.'}
          {src.kind === 'upload' && 'Processed locally; the file never leaves your browser. No model outputs exist for uploads.'}
          {src.kind === 'sample' && src.sample && `${src.sample.split} split · ${src.sample.machine ?? 'machine n/a'} · CC BY 4.0`}
        </span>
      )}
    </div>
  );
}
