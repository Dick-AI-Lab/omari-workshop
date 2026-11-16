import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { Gray, fromImageData } from '../lib/imageOps';
import { makePhantom } from '../lib/phantom';
import { loadGrayPNG, useArtifacts } from './artifacts';
import type { SampleInfo } from './types';

/**
 * One image is shared by every lab, so choosing an image in Chapter 2 also updates Chapters 3–6
 * (linked views). Sources: the synthetic phantom, an exported dataset sample, or a local upload
 * (read in the browser only, never sent anywhere).
 */
export type SourceKind = 'phantom' | 'sample' | 'upload';

interface ImageSourceState {
  kind: SourceKind;
  gray: Gray;
  name: string;
  sample?: SampleInfo;
  sectorMask?: Uint8Array;
  selectPhantom: () => void;
  selectSample: (id: string) => void;
  selectUpload: (file: File) => Promise<void>;
  loading: boolean;
}

const Ctx = createContext<ImageSourceState | null>(null);

const MAX_SIDE = 480; // keep live simulations responsive

function capSize(img: HTMLImageElement): ImageData {
  const s = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(img.naturalWidth * s));
  c.height = Math.max(1, Math.round(img.naturalHeight * s));
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, c.width, c.height);
  return ctx.getImageData(0, 0, c.width, c.height);
}

export function ImageSourceProvider({ children }: { children: ReactNode }) {
  const { manifest } = useArtifacts();
  const phantom = useMemo(() => makePhantom(320, 240, 7), []);
  const [st, setSt] = useState<{ kind: SourceKind; gray: Gray; name: string; sample?: SampleInfo; sectorMask?: Uint8Array }>({
    kind: 'phantom', gray: phantom.image, name: 'Synthetic phantom', sectorMask: phantom.sectorMask,
  });
  const [loading, setLoading] = useState(false);

  const selectPhantom = useCallback(() => setSt({ kind: 'phantom', gray: phantom.image, name: 'Synthetic phantom', sectorMask: phantom.sectorMask }), [phantom]);

  const selectSample = useCallback(async (id: string) => {
    const s = manifest?.samples.find((x) => x.id === id);
    if (!s) return;
    setLoading(true);
    try {
      const g = await loadGrayPNG(s.image);
      setSt({ kind: 'sample', gray: g, name: s.id, sample: s });
    } finally { setLoading(false); }
  }, [manifest]);

  const selectUpload = useCallback(async (file: File) => {
    setLoading(true);
    try {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.src = url;
      await img.decode();
      const g = fromImageData(capSize(img));
      URL.revokeObjectURL(url);
      setSt({ kind: 'upload', gray: g, name: file.name });
    } finally { setLoading(false); }
  }, []);

  // When an artifact bundle exists, open on the sample tagged "opening" (or the first sample).
  useEffect(() => {
    if (!manifest?.samples.length) return;
    const opening = manifest.samples.find((s) => s.tags?.includes('opening')) ?? manifest.samples[0];
    void selectSample(opening.id);
  }, [manifest, selectSample]);

  const value: ImageSourceState = { ...st, selectPhantom, selectSample, selectUpload, loading };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useImageSource(): ImageSourceState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useImageSource outside provider');
  return v;
}
