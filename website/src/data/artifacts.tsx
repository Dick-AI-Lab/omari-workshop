import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react';
import type { DatasetStats, Evaluation, Faithfulness, LabelEfficiency, Manifest, ModelId, SampleInfo, TrainingCurves } from './types';
import { CLASSES } from './literature';
import { canonicalMethod, layerLabel, methodLong } from './labels';

/** Relative to the page, so a static build works from any host and any sub-path (vite base './'). */
export const ARTIFACT_ROOT = 'experiments/';
export const SUPPORTED_SCHEMA = '1.';

type Status = 'loading' | 'absent' | 'invalid' | 'error' | 'ready';
export type FileKey = 'evaluation' | 'curves' | 'labelEfficiency' | 'datasetStats' | 'faithfulness';
/** 'absent' = not referenced or 404; 'error' = present but unreadable (bad JSON, HTTP error). */
export type FileState = 'ok' | 'absent' | 'error';

interface ArtifactState {
  status: Status;
  error?: string;
  manifest?: Manifest;
  evaluation?: Evaluation;
  curves?: TrainingCurves;
  labelEfficiency?: LabelEfficiency;
  datasetStats?: DatasetStats;
  faithfulness?: Faithfulness;
  files: Partial<Record<FileKey, { state: FileState; path?: string; message?: string }>>;
  classes: string[];
  backend: { url: string; ok: boolean; models: ModelId[] };
  hasModel: (id: ModelId) => boolean;
  samplesWith: (pred: (s: SampleInfo) => boolean) => SampleInfo[];
}

const Ctx = createContext<ArtifactState | null>(null);

export type FetchResult<T> = { kind: 'ok'; data: T } | { kind: 'absent' } | { kind: 'error'; message: string };

/**
 * Fetch a JSON artifact. Missing files are 'absent'; a file that exists but cannot be parsed
 * (e.g. bare NaN tokens) is an 'error' that is logged and surfaced, never silently treated as absent.
 */
export async function fetchJSON<T>(path: string): Promise<FetchResult<T>> {
  const url = artifactUrl(path);
  let r: Response;
  try {
    r = await fetch(url, { cache: 'no-cache' });
  } catch {
    // network failure (file:// pages, offline): nothing to read
    return { kind: 'absent' };
  }
  if (r.status === 404) return { kind: 'absent' };
  if (!r.ok) {
    const message = `${path}: HTTP ${r.status}`;
    console.error('[artifacts]', message);
    return { kind: 'error', message };
  }
  const ct = r.headers.get('content-type') ?? '';
  // Vite's dev server and many static hosts answer unknown paths with index.html; that means "absent".
  if (ct.includes('text/html')) return { kind: 'absent' };
  const text = await r.text();
  try {
    return { kind: 'ok', data: JSON.parse(text) as T };
  } catch (e) {
    const message = `${path}: invalid JSON (${(e as Error).message})`;
    console.error('[artifacts]', message);
    return { kind: 'error', message };
  }
}

function backendUrl(): string {
  const q = new URLSearchParams(window.location.search).get('backend');
  if (q) return q.replace(/\/$/, '');
  try { const s = window.localStorage.getItem('omari.backend'); if (s) return s; } catch { /* storage unavailable */ }
  return 'http://localhost:8000';
}

/** Bring a schema 1.0 manifest up to the 1.1 conventions used by the components (non-destructive). */
export function normalizeManifest(m: Manifest): Manifest {
  const samples = (m.samples ?? []).map((s) => ({
    ...s,
    attributions: s.attributions?.map((a) => {
      const method = canonicalMethod(a.method);
      return {
        ...a,
        method,
        legacy_method: method !== a.method ? a.method : a.legacy_method,
        method_label: a.method_label ?? methodLong(method),
        layer_label: a.layer_label ?? layerLabel(a.model, a.layer),
      };
    }),
    perturbations: s.perturbations?.map((p) => ({
      ...p,
      attributions: p.attributions ? Object.fromEntries(Object.entries(p.attributions).map(([k, v]) => [k, v && { ...v, method: canonicalMethod(v.method), legacy_method: canonicalMethod(v.method) !== v.method ? v.method : v.legacy_method }])) : undefined,
    })),
  }));
  return { ...m, models: m.models ?? [], samples };
}

/** Replace non-finite numbers by null in label-efficiency grids (defensive: 1.1 forbids NaN). */
function normalizeLabelEfficiency(le: LabelEfficiency): LabelEfficiency {
  const fix = (g?: (number | null)[][]) => g?.map((r) => r.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : null)));
  const per_model: LabelEfficiency['per_model'] = {};
  for (const [k, v] of Object.entries(le.per_model ?? {})) if (v) per_model[k as ModelId] = { ...v, macro_f1: fix(v.macro_f1) ?? [], accuracy: fix(v.accuracy) };
  return { ...le, per_model };
}

function normalizeFaithfulness(f: Faithfulness): Faithfulness {
  return { ...f, rows: (f.rows ?? []).map((r) => ({ ...r, method: canonicalMethod(r.method) })) };
}

export function ArtifactProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<Omit<ArtifactState, 'hasModel' | 'samplesWith' | 'classes'>>({
    status: 'loading',
    files: {},
    backend: { url: backendUrl(), ok: false, models: [] },
  });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const res = await fetchJSON<Manifest>('manifest.json');
      if (res.kind === 'error') {
        if (!cancelled) setState((s) => ({ ...s, status: 'error', error: res.message }));
      } else if (res.kind === 'absent') {
        if (!cancelled) setState((s) => ({ ...s, status: 'absent' }));
        // dataset statistics may also be exported on their own, before any model exists
        const ds = await fetchJSON<DatasetStats>('dataset_stats.json');
        if (!cancelled && ds.kind !== 'absent') setState((s) => ({ ...s, datasetStats: ds.kind === 'ok' ? ds.data : undefined, files: { ...s.files, datasetStats: ds.kind === 'ok' ? { state: 'ok', path: 'dataset_stats.json' } : { state: 'error', path: 'dataset_stats.json', message: ds.message } } }));
      } else if (!String(res.data.schema_version).startsWith(SUPPORTED_SCHEMA)) {
        if (!cancelled) setState((s) => ({ ...s, status: 'invalid', error: `Unsupported schema_version ${res.data.schema_version}` }));
      } else {
        const manifest = normalizeManifest(res.data);
        const refs: [FileKey, string | undefined][] = [
          ['evaluation', manifest.evaluation], ['curves', manifest.training_curves], ['labelEfficiency', manifest.label_efficiency],
          ['datasetStats', manifest.dataset_stats], ['faithfulness', manifest.faithfulness],
        ];
        const results = await Promise.all(refs.map(([, p]) => (p ? fetchJSON<unknown>(p) : Promise.resolve({ kind: 'absent' } as FetchResult<unknown>))));
        const files: ArtifactState['files'] = {};
        const data: Partial<Record<FileKey, unknown>> = {};
        refs.forEach(([k, p], i) => {
          const r = results[i];
          if (r.kind === 'ok') { data[k] = r.data; files[k] = { state: 'ok', path: p }; }
          else if (r.kind === 'error') files[k] = { state: 'error', path: p, message: r.message };
          else files[k] = { state: 'absent', path: p };
        });
        if (!cancelled) setState((s) => ({
          ...s, status: 'ready', manifest, files,
          evaluation: data.evaluation as Evaluation | undefined,
          curves: data.curves as TrainingCurves | undefined,
          labelEfficiency: data.labelEfficiency ? normalizeLabelEfficiency(data.labelEfficiency as LabelEfficiency) : undefined,
          datasetStats: data.datasetStats as DatasetStats | undefined,
          faithfulness: data.faithfulness ? normalizeFaithfulness(data.faithfulness as Faithfulness) : undefined,
        }));
      }
    })();
    // optional live-inference backend (python -m omari_us.serve); the site never requires it
    (async () => {
      const url = backendUrl();
      try {
        const ctl = new AbortController();
        const t = setTimeout(() => ctl.abort(), 1500);
        const r = await fetch(url + '/health', { signal: ctl.signal });
        clearTimeout(t);
        if (r.ok) {
          const j = await r.json();
          if (!cancelled) setState((s) => ({ ...s, backend: { url, ok: true, models: j.models ?? [] } }));
        }
      } catch { /* no backend: expected default */ }
    })();
    return () => { cancelled = true; };
  }, []);

  const value = useMemo<ArtifactState>(() => {
    const classes = state.manifest?.classes ?? state.datasetStats?.classes ?? CLASSES;
    return {
      ...state,
      classes,
      hasModel: (id) => !!state.manifest?.models.some((m) => m.id === id),
      samplesWith: (pred) => (state.manifest?.samples ?? []).filter(pred),
    };
  }, [state]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** Provider with preloaded data (tests and static rendering); applies the same normalisation as the loader. */
export function StaticArtifactProvider({ data, children }: { data: Partial<Pick<ArtifactState, 'manifest' | 'evaluation' | 'curves' | 'labelEfficiency' | 'datasetStats' | 'faithfulness'>>; children: ReactNode }) {
  const value = useMemo<ArtifactState>(() => {
    const manifest = data.manifest ? normalizeManifest(data.manifest) : undefined;
    const files: ArtifactState['files'] = {};
    (['evaluation', 'curves', 'labelEfficiency', 'datasetStats', 'faithfulness'] as FileKey[]).forEach((k) => { files[k] = { state: data[k] ? 'ok' : 'absent' }; });
    return {
      status: manifest ? 'ready' : 'absent', files, manifest,
      evaluation: data.evaluation, curves: data.curves, datasetStats: data.datasetStats,
      labelEfficiency: data.labelEfficiency ? normalizeLabelEfficiency(data.labelEfficiency) : undefined,
      faithfulness: data.faithfulness ? normalizeFaithfulness(data.faithfulness) : undefined,
      classes: manifest?.classes ?? CLASSES,
      backend: { url: '', ok: false, models: [] },
      hasModel: (id) => !!manifest?.models.some((m) => m.id === id),
      samplesWith: (pred) => (manifest?.samples ?? []).filter(pred),
    };
  }, [data]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useArtifacts(): ArtifactState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useArtifacts outside ArtifactProvider');
  return v;
}

export const artifactUrl = (p: string) => (/^(https?:|data:|blob:)/.test(p) ? p : ARTIFACT_ROOT + p.replace(/^\.?\//, ''));

/** Load a grayscale PNG artifact into a Float32 [0,1] array. Only called when a view needs it (lazy). */
export async function loadGrayPNG(path: string): Promise<{ w: number; h: number; data: Float32Array }> {
  const img = new Image();
  img.decoding = 'async';
  img.src = artifactUrl(path);
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const out = new Float32Array(c.width * c.height);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) out[i] = (0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2]) / 255;
  return { w: c.width, h: c.height, data: out };
}

/** Per-sample JSON (lazy). Resolves undefined when absent; rejects with an Error when present but unreadable. */
export async function loadJSON<T>(path: string): Promise<T | undefined> {
  const r = await fetchJSON<T>(path);
  if (r.kind === 'error') throw new Error(r.message);
  return r.kind === 'ok' ? r.data : undefined;
}
