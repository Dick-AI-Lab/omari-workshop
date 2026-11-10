/**
 * A synthetic B-mode *phantom* for the live simulations.
 *
 * This is NOT a patient image and NOT a FETAL_PLANES_DB sample. It is a geometric echogenicity map
 * (a sector field of view containing an elliptical bright rim, a hypoechoic interior and a few inclusions)
 * passed through a simple physical image-formation model:
 *   scatterer reflectivity × complex Gaussian (Rayleigh amplitude) speckle → anisotropic PSF blur
 *   → envelope → log compression, plus burned-in overlay text and caliper marks.
 * It exists so every preprocessing control works before any real data are loaded.
 */
import { gaussian, mulberry32 } from './rng';
import { Gray, makeGray } from './imageOps';

export interface Phantom {
  image: Gray;
  sectorMask: Uint8Array;
  annotationTruth: Uint8Array;
}

function blurSeparable(src: Float32Array, w: number, h: number, sx: number, sy: number): Float32Array {
  const kern = (s: number) => {
    const r = Math.max(1, Math.ceil(2.5 * s));
    const k: number[] = [];
    let sum = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * s * s)); k.push(v); sum += v; }
    return { k: k.map((v) => v / sum), r };
  };
  const kx = kern(sx), ky = kern(sy);
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -kx.r; i <= kx.r; i++) { const xx = Math.min(w - 1, Math.max(0, x + i)); s += src[y * w + xx] * kx.k[i + kx.r]; }
    tmp[y * w + x] = s;
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -ky.r; i <= ky.r; i++) { const yy = Math.min(h - 1, Math.max(0, y + i)); s += tmp[yy * w + x] * ky.k[i + ky.r]; }
    out[y * w + x] = s;
  }
  return out;
}

// 5×7 bitmap glyphs for the burned-in overlay
const GLYPHS: Record<string, string[]> = {
  P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
  '3': ['11110', '00001', '00001', '01110', '00001', '00001', '11110'],
  '5': ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
  '.': ['00000', '00000', '00000', '00000', '00000', '00110', '00110'],
  C: ['01110', '10001', '10000', '10000', '10000', '10001', '01110'],
};

function drawText(img: Float32Array, truth: Uint8Array, w: number, text: string, x0: number, y0: number, scale: number) {
  let cx = x0;
  for (const ch of text) {
    const g = GLYPHS[ch] ?? GLYPHS[' '];
    for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++) {
      if (g[r][c] !== '1') continue;
      for (let dy = 0; dy < scale; dy++) for (let dx = 0; dx < scale; dx++) {
        const i = (y0 + r * scale + dy) * w + cx + c * scale + dx;
        img[i] = 0.97; truth[i] = 1;
      }
    }
    cx += 6 * scale;
  }
}

export function makePhantom(width = 320, height = 240, seed = 7): Phantom {
  const rand = mulberry32(seed);
  const w = width, h = height;
  const refl = new Float32Array(w * h);
  const mask = new Uint8Array(w * h);
  // sector geometry (curvilinear probe at top centre)
  const apexX = w / 2, apexY = -h * 0.18;
  const rMin = h * 0.27, rMax = h * 1.12;
  const halfAngle = (36 * Math.PI) / 180;
  // anatomy-agnostic geometric "target": bright elliptical rim with darker interior and inclusions
  const cx = w * 0.5, cy = h * 0.56, ax = w * 0.22, ay = h * 0.26, rot = -0.25;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = x - apexX, dy = y - apexY;
    const r = Math.hypot(dx, dy), th = Math.atan2(dx, dy);
    if (r < rMin || r > rMax || Math.abs(th) > halfAngle) continue;
    mask[y * w + x] = 1;
    // background tissue echogenicity with depth attenuation
    let e = 0.35 + 0.08 * Math.sin(x * 0.05) * Math.cos(y * 0.04);
    const ux = ((x - cx) * Math.cos(rot) + (y - cy) * Math.sin(rot)) / ax;
    const uy = (-(x - cx) * Math.sin(rot) + (y - cy) * Math.cos(rot)) / ay;
    const q = Math.sqrt(ux * ux + uy * uy);
    if (q < 0.92) e = 0.12 + 0.05 * q; // hypoechoic interior
    if (q >= 0.92 && q < 1.04) e = 1.0; // bright rim
    if (Math.hypot(ux - 0.15, uy + 0.1) < 0.18) e = 0.55; // inclusion
    if (Math.abs(uy - 0.05) < 0.035 && Math.abs(ux) < 0.55) e = 0.7; // midline echo
    const depth = (r - rMin) / (rMax - rMin);
    e *= Math.exp(-1.1 * depth);
    // posterior acoustic shadow behind the bright rim's lower edge
    if (q > 1.04 && y > cy + ay * 0.8 && Math.abs(x - cx) < ax * 0.35) e *= 0.45;
    refl[y * w + x] = e;
  }
  // complex Gaussian scattering → envelope after PSF blur (axial short, lateral wider)
  const re = new Float32Array(w * h), im = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const a = Math.sqrt(refl[i]);
    re[i] = a * gaussian(rand);
    im[i] = a * gaussian(rand);
  }
  const reB = blurSeparable(re, w, h, 1.6, 0.9);
  const imB = blurSeparable(im, w, h, 1.6, 0.9);
  const env = new Float32Array(w * h);
  let mx = 0;
  for (let i = 0; i < w * h; i++) { env[i] = Math.hypot(reB[i], imB[i]); if (env[i] > mx) mx = env[i]; }
  const img = makeGray(w, h);
  const dyn = 48; // dB dynamic range
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) { img.data[i] = 0; continue; }
    const db = 20 * Math.log10(env[i] / mx + 1e-6);
    img.data[i] = Math.max(0, Math.min(1, (db + dyn) / dyn));
  }
  const truth = new Uint8Array(w * h);
  drawText(img.data, truth, w, 'PHANTOM', 8, 8, 1);
  drawText(img.data, truth, w, 'C5 3.5', w - 46, 8, 1);
  // caliper marks "+"
  const caliper = (px: number, py: number) => {
    for (let k = -4; k <= 4; k++) {
      img.data[py * w + px + k] = 0.97; truth[py * w + px + k] = 1;
      img.data[(py + k) * w + px] = 0.97; truth[(py + k) * w + px] = 1;
    }
  };
  caliper(Math.round(cx - ax * 0.98), Math.round(cy + 8));
  caliper(Math.round(cx + ax * 0.98), Math.round(cy - 14));
  return { image: img, sectorMask: mask, annotationTruth: truth };
}
