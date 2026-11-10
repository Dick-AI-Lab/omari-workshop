import { createContext, ReactNode, useCallback, useContext, useEffect, useRef, useState } from 'react';

export interface ChapterDef { id: string; num?: number; title: string; short: string; start: number; end: number; cue: string; }

export const CHAPTERS: ChapterDef[] = [
  { id: 'opening', num: 1, title: 'Can AI recognise this ultrasound plane?', short: 'Opening', start: 0, end: 10, cue: 'Poll the room before revealing. Separate correctness, confidence and reasoning.' },
  { id: 'data', num: 2, title: 'From ultrasound pixels to training data', short: 'Data', start: 10, end: 25, cue: 'Speckle → normalisation → resize → imbalance → leakage. Spend 4 min on splitting.' },
  { id: 'models', num: 3, title: 'Inside the models', short: 'Models', start: 25, end: 50, cue: '3A DenseNet 9 min · 3B ViT 9 min · 3C USF-MAE 7 min.' },
  { id: 'evaluation', num: 4, title: 'Training, evaluation and generalisation', short: 'Evaluation', start: 50, end: 70, cue: 'Confusion → per-class → prevalence → calibration → label efficiency.' },
  { id: 'break', title: 'Break', short: 'Break', start: 70, end: 80, cue: 'Leave the explainability lab open for exploration.' },
  { id: 'explain', num: 5, title: 'Explainability: from CAM to MetaCAM++', short: 'Explainability', start: 80, end: 105, cue: 'Toy CAM mechanics first (8 min), then real maps (12 min), then ViT caveats (5 min).' },
  { id: 'detective', num: 6, title: 'Model detective: can we trust the explanation?', short: 'Detective', start: 105, end: 115, cue: 'Hypothesis first, evidence second. Plausible ≠ faithful.' },
  { id: 'closing', num: 7, title: 'Closing: what would you trust?', short: 'Closing', start: 115, end: 120, cue: 'Return to the opening image. Collect answers to the deployment question.' },
];
export const TOTAL_MIN = 120;

interface PresenterState {
  presenting: boolean;
  toggle: () => void;
  elapsed: number;
  running: boolean;
  setRunning: (v: boolean) => void;
  reset: () => void;
  current: string;
}
const Ctx = createContext<PresenterState | null>(null);
export const usePresenter = () => useContext(Ctx)!;

export function PresenterProvider({ children }: { children: ReactNode }) {
  const [presenting, setPresenting] = useState(() => new URLSearchParams(location.search).get('mode') === 'present');
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const [current, setCurrent] = useState(CHAPTERS[0].id);
  const last = useRef<number | null>(null);

  useEffect(() => { document.body.classList.toggle('presenting', presenting); }, [presenting]);

  useEffect(() => {
    if (!running) { last.current = null; return; }
    const id = window.setInterval(() => {
      const now = performance.now();
      if (last.current != null) setElapsed((e) => e + (now - last.current!) / 1000);
      last.current = now;
    }, 500);
    return () => window.clearInterval(id);
  }, [running]);

  // which chapter is on screen
  useEffect(() => {
    const els = CHAPTERS.map((c) => document.getElementById(c.id)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (vis[0]) setCurrent(vis[0].target.id);
    }, { rootMargin: '-40% 0px -55% 0px' });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);

  const toggle = useCallback(() => setPresenting((p) => !p), []);

  // keyboard navigation: ←/→ or PgUp/PgDn step through figures and headings ("beats"); [ and ] jump chapters
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const beats = Array.from(document.querySelectorAll<HTMLElement>('[data-beat], .chapter'));
      const y = window.scrollY + 80;
      const go = (el?: HTMLElement) => el && window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 70, behavior: 'smooth' });
      if (e.key === 'ArrowRight' || e.key === 'PageDown' || (e.key === ' ' && presenting)) {
        e.preventDefault(); go(beats.find((b) => b.getBoundingClientRect().top + window.scrollY > y + 10));
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault(); go([...beats].reverse().find((b) => b.getBoundingClientRect().top + window.scrollY < y - 10));
      } else if (e.key === ']' || e.key === '[') {
        const idx = CHAPTERS.findIndex((c) => c.id === current);
        const next = CHAPTERS[Math.max(0, Math.min(CHAPTERS.length - 1, idx + (e.key === ']' ? 1 : -1)))];
        go(document.getElementById(next.id) ?? undefined);
      } else if (e.key === 'p' || e.key === 'P') toggle();
      else if (e.key === 'f' || e.key === 'F') {
        if (document.fullscreenElement) void document.exitFullscreen(); else void document.documentElement.requestFullscreen?.();
      } else if (e.key === 't' || e.key === 'T') setRunning((r) => !r);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [presenting, current, toggle]);

  return <Ctx.Provider value={{ presenting, toggle, elapsed, running, setRunning, reset: () => setElapsed(0), current }}>{children}</Ctx.Provider>;
}

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export function TopBar() {
  const p = usePresenter();
  const elapsedMin = p.elapsed / 60;
  const scrollMin = (() => {
    const c = CHAPTERS.find((x) => x.id === p.current);
    return c ? c.start : 0;
  })();
  const clockMin = p.running || p.elapsed > 0 ? elapsedMin : null;
  return (
    <header className="topbar">
      <a className="brand" href="#top">Pixels → Predictions <span>· OMARI</span></a>
      <nav className="timeline" aria-label="Workshop chapters, sized by minutes">
        {CHAPTERS.map((c) => {
          const fill = Math.max(0, Math.min(1, ((clockMin ?? scrollMin) - c.start) / (c.end - c.start)));
          return (
            <a key={c.id} href={`#${c.id}`} className={c.id === 'break' ? 'break' : ''} aria-current={p.current === c.id}
              style={{ flex: `${c.end - c.start} 1 0` }} title={`${c.num ? c.num + '. ' : ''}${c.title} (${c.start}–${c.end} min)`}>
              <span className="fill" style={{ width: `${fill * 100}%` }} />
              <span className="t">{c.num ? `${c.num} ` : ''}{c.short}</span>
            </a>
          );
        })}
        {clockMin != null && (
          <span className="clock" style={{ left: `${Math.min(100, (clockMin / TOTAL_MIN) * 100)}%` }} data-label={mmss(p.elapsed)} aria-hidden />
        )}
      </nav>
      <div className="bar-actions">
        <button className="btn" onClick={p.toggle} aria-pressed={p.presenting} title="Toggle presenter mode (P)">{p.presenting ? 'Participant view' : 'Present'}</button>
      </div>
    </header>
  );
}

export function PresenterHUD() {
  const p = usePresenter();
  if (!p.presenting) return null;
  const c = CHAPTERS.find((x) => x.id === p.current) ?? CHAPTERS[0];
  const min = p.elapsed / 60;
  const over = min > c.end;
  return (
    <aside className="presenter-hud" aria-label="Presenter controls">
      <div className="row"><b>{c.num ? `Chapter ${c.num}` : c.title}</b><span>{c.start}–{c.end} min</span></div>
      <div className="row"><span className={over ? 'over' : ''}>{mmss(p.elapsed)} elapsed{over ? ` · ${Math.ceil(min - c.end)} min over` : ''}</span>
        <span className="row" style={{ gap: '0.3rem' }}>
          <button onClick={() => p.setRunning(!p.running)}>{p.running ? 'Pause' : 'Start'}</button>
          <button onClick={p.reset}>Reset</button>
        </span>
      </div>
      <div className="beat">{c.cue}</div>
      <div className="beat">← → step · [ ] chapter · F fullscreen · T timer · P exit</div>
    </aside>
  );
}

export function ChapterHead({ id }: { id: string }) {
  const c = CHAPTERS.find((x) => x.id === id)!;
  return (
    <div className="chapter-head">
      <div className="chapter-num" aria-hidden>{c.num ?? '—'}</div>
      <h2>{c.title}</h2>
      <div className="chapter-time">{c.start}–{c.end} min · {c.end - c.start} minutes</div>
    </div>
  );
}
