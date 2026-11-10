/**
 * Pure, dependency-free grayscale image operations used by the live simulations.
 * Images are Float32Array in [0, 1] (unless stated), row-major, width × height.
 * Nothing here involves a trained model.
 */
import { mulberry32, speckleSample } from './rng';

export interface Gray {
  w: number;
  h: number;
  data: Float32Array;
}

export const makeGray = (w: number, h: number, fill = 0): Gray => ({ w, h, data: new Float32Array(w * h).fill(fill) });
export const cloneGray = (g: Gray): Gray => ({ w: g.w, h: g.h, data: new Float32Array(g.data) });

export function fromImageData(img: ImageData): Gray {
  const out = makeGray(img.width, img.height);
  const d = img.data;
  for (let i = 0, j = 0; i < out.data.length; i++, j += 4) {
    // ITU-R BT.601 luma; FETAL_PLANES_DB PNGs are grayscale so R=G=B in practice.
    out.data[i] = (0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2]) / 255;
  }
  return out;
}

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Brightness (additive), contrast (gain about mid-grey 0.5) and gamma. */
export function adjust(g: Gray, brightness: number, contrast: number, gamma = 1): Gray {
  const out = makeGray(g.w, g.h);
  for (let i = 0; i < g.data.length; i++) {
    let v = (g.data[i] - 0.5) * contrast + 0.5 + brightness;
    v = clamp01(v);
    out.data[i] = gamma === 1 ? v : Math.pow(v, gamma);
  }
  return out;
}

export function sampleBilinear(g: Gray, x: number, y: number): number {
  const x0 = Math.max(0, Math.min(g.w - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(g.h - 1, Math.floor(y)));
  const x1 = Math.min(g.w - 1, x0 + 1);
  const y1 = Math.min(g.h - 1, y0 + 1);
  const fx = Math.max(0, Math.min(1, x - x0));
  const fy = Math.max(0, Math.min(1, y - y0));
  const a = g.data[y0 * g.w + x0];
  const b = g.data[y0 * g.w + x1];
  const c = g.data[y1 * g.w + x0];
  const d = g.data[y1 * g.w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

/** Resize with bilinear or nearest interpolation (pixel-centre aligned, like align_corners=False). */
export function resize(g: Gray, w: number, h: number, mode: 'bilinear' | 'nearest' | 'area' = 'bilinear'): Gray {
  const out = makeGray(w, h);
  const sx = g.w / w;
  const sy = g.h / h;
  if (mode === 'area' && sx >= 1 && sy >= 1) {
    // box filter (antialiased downsampling)
    for (let y = 0; y < h; y++) {
      const y0 = Math.floor(y * sy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
      for (let x = 0; x < w; x++) {
        const x0 = Math.floor(x * sx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
        let s = 0, n = 0;
        for (let yy = y0; yy < y1 && yy < g.h; yy++) for (let xx = x0; xx < x1 && xx < g.w; xx++) { s += g.data[yy * g.w + xx]; n++; }
        out.data[y * w + x] = n ? s / n : 0;
      }
    }
    return out;
  }
  for (let y = 0; y < h; y++) {
    const srcY = (y + 0.5) * sy - 0.5;
    for (let x = 0; x < w; x++) {
      const srcX = (x + 0.5) * sx - 0.5;
      out.data[y * w + x] =
        mode === 'nearest'
          ? g.data[Math.min(g.h - 1, Math.max(0, Math.round(srcY))) * g.w + Math.min(g.w - 1, Math.max(0, Math.round(srcX)))]
          : sampleBilinear(g, srcX, srcY);
    }
  }
  return out;
}

/** Simulate a lower acquisition / storage resolution: downsample (area) then upsample back. */
export function simulateResolution(g: Gray, factor: number, upMode: 'bilinear' | 'nearest' = 'nearest'): Gray {
  if (factor <= 1) return cloneGray(g);
  const w = Math.max(2, Math.round(g.w / factor));
  const h = Math.max(2, Math.round(g.h / factor));
  return resize(resize(g, w, h, 'area'), g.w, g.h, upMode);
}

/**
 * Multiplicative speckle: I' = I · (1 + s·(n − 1)), n ~ Gamma(L, 1/L).
 * Fully developed speckle intensity follows an exponential law (L = 1); averaging L looks reduces variance to 1/L.
 * `mask` (optional) restricts the noise to the imaging sector.
 */
export function addSpeckle(g: Gray, strength: number, looks: number, seed: number, mask?: Uint8Array): Gray {
  const rand = mulberry32(seed);
  const out = makeGray(g.w, g.h);
  for (let i = 0; i < g.data.length; i++) {
    const n = speckleSample(rand, looks);
    if (mask && !mask[i]) { out.data[i] = g.data[i]; continue; }
    out.data[i] = clamp01(g.data[i] * (1 + strength * (n - 1)));
  }
  return out;
}

export function histogram(g: Gray, bins = 64, lo = 0, hi = 1): number[] {
  const h = new Array(bins).fill(0);
  const span = hi - lo || 1;
  for (let i = 0; i < g.data.length; i++) {
    let b = Math.floor(((g.data[i] - lo) / span) * bins);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    h[b]++;
  }
  return h;
}

export function stats(data: Float32Array | number[]): { mean: number; std: number; min: number; max: number } {
  let s = 0, s2 = 0, mn = Infinity, mx = -Infinity;
  const n = data.length;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    s += v; s2 += v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const mean = s / n;
  return { mean, std: Math.sqrt(Math.max(0, s2 / n - mean * mean)), min: mn, max: mx };
}

export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

export type NormMethod = 'none' | 'minmax' | 'zscore' | 'imagenet' | 'half' | 'histeq' | 'clahe';

export interface Normalized {
  /** values actually fed to the network (may be outside [0,1]) */
  values: Float32Array;
  /** per-channel tensors when the method replicates grey → RGB (ImageNet / 0.5 conventions) */
  channels?: Float32Array[];
  /** a [0,1] version for display */
  display: Gray;
  range: [number, number];
}

/** Normalisation methods participants can compare. 'imagenet' replicates grey into 3 channels then standardises each. */
export function normalize(g: Gray, method: NormMethod): Normalized {
  const n = g.data.length;
  const disp = (vals: Float32Array, lo: number, hi: number): Gray => {
    const o = makeGray(g.w, g.h);
    const span = hi - lo || 1;
    for (let i = 0; i < n; i++) o.data[i] = clamp01((vals[i] - lo) / span);
    return o;
  };
  if (method === 'none') return { values: new Float32Array(g.data), display: cloneGray(g), range: [0, 1] };
  if (method === 'minmax') {
    const s = stats(g.data);
    const v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = (g.data[i] - s.min) / (s.max - s.min || 1);
    return { values: v, display: disp(v, 0, 1), range: [0, 1] };
  }
  if (method === 'zscore') {
    const s = stats(g.data);
    const v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = (g.data[i] - s.mean) / (s.std || 1);
    return { values: v, display: disp(v, -3, 3), range: [-3, 3] };
  }
  if (method === 'imagenet' || method === 'half') {
    const mean = method === 'imagenet' ? IMAGENET_MEAN : [0.5, 0.5, 0.5];
    const std = method === 'imagenet' ? IMAGENET_STD : [0.5, 0.5, 0.5];
    const ch = [0, 1, 2].map((c) => {
      const v = new Float32Array(n);
      for (let i = 0; i < n; i++) v[i] = (g.data[i] - mean[c]) / std[c];
      return v;
    });
    const lo = (0 - Math.max(...mean)) / Math.min(...std);
    const hi = (1 - Math.min(...mean)) / Math.min(...std);
    return { values: ch[0], channels: ch, display: disp(ch[0], lo, hi), range: [lo, hi] };
  }
  if (method === 'histeq') {
    const out = equalize(g);
    return { values: out.data, display: out, range: [0, 1] };
  }
  const out = clahe(g, 8, 2.0);
  return { values: out.data, display: out, range: [0, 1] };
}

/** Global histogram equalisation (256 bins). */
export function equalize(g: Gray): Gray {
  const bins = 256;
  const h = histogram(g, bins);
  const cdf = new Array(bins).fill(0);
  let acc = 0;
  for (let i = 0; i < bins; i++) { acc += h[i]; cdf[i] = acc; }
  const cmin = cdf.find((c) => c > 0) ?? 0;
  const total = g.data.length;
  const out = makeGray(g.w, g.h);
  for (let i = 0; i < total; i++) {
    const b = Math.min(bins - 1, Math.floor(g.data[i] * bins));
    out.data[i] = (cdf[b] - cmin) / (total - cmin || 1);
  }
  return out;
}

/**
 * Contrast-limited adaptive histogram equalisation (Zuiderveld 1994), tiles × tiles grid,
 * clip limit relative to the uniform bin height, bilinear interpolation between tile mappings.
 */
export function clahe(g: Gray, tiles = 8, clipLimit = 2.0): Gray {
  const bins = 256;
  const tw = g.w / tiles, th = g.h / tiles;
  const maps: Float32Array[] = [];
  for (let ty = 0; ty < tiles; ty++) {
    for (let tx = 0; tx < tiles; tx++) {
      const hist = new Float32Array(bins);
      const x0 = Math.floor(tx * tw), x1 = Math.floor((tx + 1) * tw);
      const y0 = Math.floor(ty * th), y1 = Math.floor((ty + 1) * th);
      let count = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        hist[Math.min(bins - 1, Math.floor(g.data[y * g.w + x] * bins))]++; count++;
      }
      const limit = Math.max(1, (clipLimit * count) / bins);
      let excess = 0;
      for (let b = 0; b < bins; b++) if (hist[b] > limit) { excess += hist[b] - limit; hist[b] = limit; }
      const add = excess / bins;
      const map = new Float32Array(bins);
      let acc = 0;
      for (let b = 0; b < bins; b++) { acc += hist[b] + add; map[b] = acc / (count || 1); }
      maps.push(map);
    }
  }
  const out = makeGray(g.w, g.h);
  for (let y = 0; y < g.h; y++) {
    const gy = y / th - 0.5;
    const ty0 = Math.max(0, Math.min(tiles - 1, Math.floor(gy)));
    const ty1 = Math.min(tiles - 1, ty0 + 1);
    const fy = Math.max(0, Math.min(1, gy - ty0));
    for (let x = 0; x < g.w; x++) {
      const gx = x / tw - 0.5;
      const tx0 = Math.max(0, Math.min(tiles - 1, Math.floor(gx)));
      const tx1 = Math.min(tiles - 1, tx0 + 1);
      const fx = Math.max(0, Math.min(1, gx - tx0));
      const b = Math.min(bins - 1, Math.floor(g.data[y * g.w + x] * bins));
      const a = maps[ty0 * tiles + tx0][b], bb = maps[ty0 * tiles + tx1][b];
      const c = maps[ty1 * tiles + tx0][b], d = maps[ty1 * tiles + tx1][b];
      out.data[y * g.w + x] = (a * (1 - fx) + bb * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    }
  }
  return out;
}

export type FitMode = 'stretch' | 'letterbox' | 'centercrop';

export interface FitResult {
  out: Gray;
  /** where the source image lands inside the output square (letterbox) or which source region is kept (crop) */
  content: { x: number; y: number; w: number; h: number };
  scaleX: number;
  scaleY: number;
}

/**
 * Map an arbitrary W×H image to a size×size network input.
 *  - stretch: independent x/y scaling (what the public USF-MAE loader does: Resize((224,224), NEAREST))
 *  - letterbox: preserve aspect ratio, pad with zeros
 *  - centercrop: resize shorter side to `size`, crop the centre (torchvision Resize(int)+CenterCrop convention)
 */
export function fitToSquare(g: Gray, size: number, mode: FitMode, interp: 'bilinear' | 'nearest' = 'bilinear'): FitResult {
  if (mode === 'stretch') {
    return { out: resize(g, size, size, interp), content: { x: 0, y: 0, w: size, h: size }, scaleX: size / g.w, scaleY: size / g.h };
  }
  if (mode === 'letterbox') {
    const s = Math.min(size / g.w, size / g.h);
    const w = Math.max(1, Math.round(g.w * s)), h = Math.max(1, Math.round(g.h * s));
    const r = resize(g, w, h, interp);
    const out = makeGray(size, size, 0);
    const ox = Math.floor((size - w) / 2), oy = Math.floor((size - h) / 2);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out.data[(y + oy) * size + x + ox] = r.data[y * w + x];
    return { out, content: { x: ox, y: oy, w, h }, scaleX: s, scaleY: s };
  }
  const s = Math.max(size / g.w, size / g.h);
  const w = Math.max(size, Math.round(g.w * s)), h = Math.max(size, Math.round(g.h * s));
  const r = resize(g, w, h, interp);
  const out = makeGray(size, size);
  const ox = Math.floor((w - size) / 2), oy = Math.floor((h - size) / 2);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) out.data[y * size + x] = r.data[(y + oy) * w + x + ox];
  // content = the region of the *source* that survives, in source pixel units
  return { out, content: { x: ox / s, y: oy / s, w: size / s, h: size / s }, scaleX: s, scaleY: s };
}

export interface ConvSpec {
  kernel: number[][];
  stride: number;
  padding: number;
  dilation: number;
}

/** Single-channel 2D cross-correlation (what deep-learning libraries call convolution) with zero padding. */
export function conv2d(g: Gray, spec: ConvSpec): Gray {
  const { kernel, stride, padding, dilation } = spec;
  const kh = kernel.length, kw = kernel[0].length;
  const effH = dilation * (kh - 1) + 1, effW = dilation * (kw - 1) + 1;
  const oh = Math.floor((g.h + 2 * padding - effH) / stride) + 1;
  const ow = Math.floor((g.w + 2 * padding - effW) / stride) + 1;
  const out = makeGray(Math.max(1, ow), Math.max(1, oh));
  for (let oy = 0; oy < oh; oy++) {
    for (let ox = 0; ox < ow; ox++) {
      let s = 0;
      for (let ky = 0; ky < kh; ky++) {
        const iy = oy * stride - padding + ky * dilation;
        if (iy < 0 || iy >= g.h) continue;
        for (let kx = 0; kx < kw; kx++) {
          const ix = ox * stride - padding + kx * dilation;
          if (ix < 0 || ix >= g.w) continue;
          s += kernel[ky][kx] * g.data[iy * g.w + ix];
        }
      }
      out.data[oy * ow + ox] = s;
    }
  }
  return out;
}

export function relu(g: Gray): Gray {
  const o = makeGray(g.w, g.h);
  for (let i = 0; i < g.data.length; i++) o.data[i] = g.data[i] > 0 ? g.data[i] : 0;
  return o;
}

export function maxPool(g: Gray, k = 2, stride = 2): Gray {
  const ow = Math.floor((g.w - k) / stride) + 1, oh = Math.floor((g.h - k) / stride) + 1;
  const o = makeGray(ow, oh);
  for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
    let m = -Infinity;
    for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) m = Math.max(m, g.data[(y * stride + dy) * g.w + x * stride + dx]);
    o.data[y * ow + x] = m;
  }
  return o;
}

/** Rescale arbitrary values to [0,1] for display; symmetric=true maps 0 → 0.5 (for signed feature maps). */
export function toDisplay(g: Gray, symmetric = false): Gray {
  const s = stats(g.data);
  const o = makeGray(g.w, g.h);
  if (symmetric) {
    const m = Math.max(Math.abs(s.min), Math.abs(s.max)) || 1;
    for (let i = 0; i < g.data.length; i++) o.data[i] = 0.5 + g.data[i] / (2 * m);
  } else {
    const span = s.max - s.min || 1;
    for (let i = 0; i < g.data.length; i++) o.data[i] = (g.data[i] - s.min) / span;
  }
  return o;
}

/** Mean intensity of each p×p patch on a grid (used by the patch / masking explorers). */
export function patchMeans(g: Gray, p: number): Float32Array {
  const gw = Math.floor(g.w / p), gh = Math.floor(g.h / p);
  const out = new Float32Array(gw * gh);
  for (let py = 0; py < gh; py++) for (let px = 0; px < gw; px++) {
    let s = 0;
    for (let y = 0; y < p; y++) for (let x = 0; x < p; x++) s += g.data[(py * p + y) * g.w + px * p + x];
    out[py * gw + px] = s / (p * p);
  }
  return out;
}

/** Bright, near-saturated, low-texture pixels — a crude detector for burned-in text, calipers and markers. */
export function annotationMask(g: Gray, thresh = 0.92, dilate = 1): Uint8Array {
  const m = new Uint8Array(g.w * g.h);
  for (let i = 0; i < m.length; i++) m[i] = g.data[i] >= thresh ? 1 : 0;
  if (dilate <= 0) return m;
  const o = new Uint8Array(m.length);
  for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
    let on = 0;
    for (let dy = -dilate; dy <= dilate && !on; dy++) for (let dx = -dilate; dx <= dilate; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < g.h && xx >= 0 && xx < g.w && m[yy * g.w + xx]) { on = 1; break; }
    }
    o[y * g.w + x] = on;
  }
  return o;
}
