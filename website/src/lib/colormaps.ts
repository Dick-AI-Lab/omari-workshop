import { Gray } from './imageOps';

export type ColormapName = 'gray' | 'inferno' | 'viridis' | 'jet' | 'diverging';

// 9-stop approximations of matplotlib's perceptually uniform maps (inferno, viridis).
const STOPS: Record<Exclude<ColormapName, 'gray'>, [number, number, number][]> = {
  inferno: [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]],
  viridis: [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [253, 231, 37]],
  // jet is included only so participants can see why it is a poor choice (non-monotone luminance)
  jet: [[0, 0, 131], [0, 0, 255], [0, 128, 255], [0, 255, 255], [128, 255, 128], [255, 255, 0], [255, 128, 0], [255, 0, 0], [128, 0, 0]],
  diverging: [[33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240], [247, 247, 247], [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43]],
};

export function colormap(name: ColormapName, t: number): [number, number, number] {
  const v = t < 0 ? 0 : t > 1 ? 1 : t;
  if (name === 'gray') { const g = Math.round(v * 255); return [g, g, g]; }
  const s = STOPS[name];
  const f = v * (s.length - 1);
  const i = Math.min(s.length - 2, Math.floor(f));
  const u = f - i;
  return [0, 1, 2].map((c) => Math.round(s[i][c] * (1 - u) + s[i + 1][c] * u)) as [number, number, number];
}

export function grayToImageData(g: Gray, cmap: ColormapName = 'gray'): ImageData {
  const id = new ImageData(g.w, g.h);
  for (let i = 0, j = 0; i < g.data.length; i++, j += 4) {
    const [r, gg, b] = colormap(cmap, g.data[i]);
    id.data[j] = r; id.data[j + 1] = gg; id.data[j + 2] = b; id.data[j + 3] = 255;
  }
  return id;
}

/** Alpha-blend a [0,1] heatmap (any resolution, bilinearly upsampled) over a grey base image. */
export function overlayImageData(base: Gray, heat: Gray, opacity: number, cmap: ColormapName, threshold = 0): ImageData {
  const id = new ImageData(base.w, base.h);
  const sx = heat.w / base.w, sy = heat.h / base.h;
  for (let y = 0; y < base.h; y++) {
    for (let x = 0; x < base.w; x++) {
      const i = y * base.w + x, j = i * 4;
      const hx = Math.min(heat.w - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const hy = Math.min(heat.h - 1, Math.max(0, (y + 0.5) * sy - 0.5));
      const x0 = Math.floor(hx), y0 = Math.floor(hy), x1 = Math.min(heat.w - 1, x0 + 1), y1 = Math.min(heat.h - 1, y0 + 1);
      const fx = hx - x0, fy = hy - y0;
      const hv = (heat.data[y0 * heat.w + x0] * (1 - fx) + heat.data[y0 * heat.w + x1] * fx) * (1 - fy) +
        (heat.data[y1 * heat.w + x0] * (1 - fx) + heat.data[y1 * heat.w + x1] * fx) * fy;
      const g = Math.round(base.data[i] * 255);
      const a = hv >= threshold ? opacity : 0;
      const [r, gg, b] = colormap(cmap, hv);
      id.data[j] = Math.round(g * (1 - a) + r * a);
      id.data[j + 1] = Math.round(g * (1 - a) + gg * a);
      id.data[j + 2] = Math.round(g * (1 - a) + b * a);
      id.data[j + 3] = 255;
    }
  }
  return id;
}

export function cssGradient(name: ColormapName): string {
  const stops = Array.from({ length: 9 }, (_, i) => {
    const [r, g, b] = colormap(name, i / 8);
    return `rgb(${r},${g},${b}) ${(i / 8) * 100}%`;
  });
  return `linear-gradient(90deg, ${stops.join(',')})`;
}
