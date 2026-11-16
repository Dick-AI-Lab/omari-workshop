/**
 * Smoke test: server-render every chapter against (a) no bundle, (b) the installed schema 1.1 bundle in
 * public/experiments, (c) the original schema 1.0 bundle under content/ (read only). Bundles that are not
 * present on this machine are skipped. Catches runtime errors in data handling without a browser.
 */
import { createElement as h } from 'react';
import { renderToString } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { StaticArtifactProvider } from '../data/artifacts';
import { ImageSourceProvider } from '../data/imageSource';
import Ch1Opening from '../chapters/Ch1Opening';
import Ch2Data from '../chapters/Ch2Data';
import Ch3Models from '../chapters/Ch3Models';
import Ch4Evaluation from '../chapters/Ch4Evaluation';
import Ch5Explain from '../chapters/Ch5Explain';
import { Ch6Detective, Ch7Closing } from '../chapters/Ch6to7';

type Json = Record<string, unknown>;
// import.meta.glob keeps this test free of Node typings; a missing directory just yields no files
const v11Files = import.meta.glob('../../public/experiments/*.json', { eager: true, import: 'default' }) as Record<string, Json>;
const v10Files = import.meta.glob('../../../content/omari_artifacts/omari_artifacts/website_bundle/*.json', { eager: true, import: 'default' }) as Record<string, Json>;
function loadBundle(files: Record<string, Json>) {
  const read = (name?: unknown) => (typeof name === 'string' ? Object.entries(files).find(([k]) => k.endsWith('/' + name))?.[1] : undefined) as never;
  const manifest = read('manifest.json') as Json | undefined;
  return {
    manifest: manifest as never, evaluation: read(manifest?.evaluation), curves: read(manifest?.training_curves), labelEfficiency: read(manifest?.label_efficiency),
    datasetStats: read(manifest?.dataset_stats), faithfulness: read(manifest?.faithfulness),
  };
}
const hasManifest = (f: Record<string, Json>) => Object.keys(f).some((k) => k.endsWith('/manifest.json'));
const chapters = [Ch1Opening, Ch2Data, Ch3Models, Ch4Evaluation, Ch5Explain, Ch6Detective, Ch7Closing];
const renderAll = (data: Parameters<typeof StaticArtifactProvider>[0]['data']) =>
  chapters.map((C) => renderToString(h(StaticArtifactProvider, { data, children: h(ImageSourceProvider, { children: h(C) }) })));

describe('chapters render', () => {
  it('without any bundle (empty states)', () => {
    const html = renderAll({}).join('');
    expect(html).toContain('No exported samples yet');
    expect(html).toContain('No label-efficiency runs exported');
  });

  it.skipIf(!hasManifest(v11Files))('with the installed schema 1.1 bundle', () => {
    const data = loadBundle(v11Files);
    const html = renderAll(data).join('');
    expect(html).not.toContain('NaN');
    expect(html).toContain('Deletion AUC');
    expect(html).toContain('not run');
    expect(html).toContain('usf_mae_500ep.pth');
    expect(html).not.toMatch(/usf_mae_100ep\.pth[^<]*\(used for this workshop\)/);
  });

  it.skipIf(!hasManifest(v10Files))('with the original schema 1.0 bundle', () => {
    const html = renderAll(loadBundle(v10Files)).join('');
    expect(html).not.toContain('NaN');
    expect(html).toContain('Test-set performance');
  });
});
