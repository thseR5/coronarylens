import { useEffect, useRef, useState } from "react";
import { useStore } from "../store";
import {
  ARTERY_COLORS,
  cssGradient,
  NEUTRAL_VESSEL,
  NOT_MODELLED,
} from "./colorScale";

const STRIPES = `repeating-linear-gradient(45deg, #7ad151 0 4px, #0b1118 4px 7px)`;

/** Below this scene width the full legend would cover the lower vessel labels, so it starts collapsed. */
export const COMPACT_LEGEND_BELOW_PX = 900;

/** Colour-scale legend with each vessel's operating threshold marked. */
export default function Legend({ className = "" }: { className?: string }) {
  const showTerritories = useStore((s) => s.view.showTerritories);
  const analysis = useStore((s) => s.analysis);
  const mode = useStore((s) => s.mode);
  const vessels = analysis ? Object.entries(analysis.vessels) : [];
  const ref = useRef<HTMLDivElement>(null);
  // null = follow the scene width; true/false = the user's explicit choice
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const parent = ref.current?.parentElement;
    if (!parent) return;
    const update = () =>
      setNarrow(parent.clientWidth < COMPACT_LEGEND_BELOW_PX);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);
  const open = userOpen ?? !narrow;

  if (!open)
    return (
      <div
        ref={ref}
        className={`rounded-md border border-line bg-panel/90 px-2 py-1 text-[10px] text-muted backdrop-blur ${className}`}
        data-testid="legend"
        data-compact="true"
      >
        <div className="flex items-center gap-2">
          <span className="text-text">P(stenosis)</span>
          <span className="font-mono">0%</span>
          <span
            className="h-2.5 w-24 rounded-sm"
            style={{ background: cssGradient() }}
            aria-hidden
          />
          <span className="font-mono">100%</span>
          <span
            className="inline-block h-2.5 w-4 rounded-sm"
            style={{ background: STRIPES }}
            title="uncertain (conformal)"
          />
          <button
            type="button"
            className="text-accent underline"
            onClick={() => setUserOpen(true)}
            aria-expanded={false}
          >
            Legend
          </button>
        </div>
      </div>
    );

  return (
    <div
      ref={ref}
      className={`w-60 rounded-md border border-line bg-panel/90 p-2 text-[11px] text-muted backdrop-blur ${className}`}
      data-testid="legend"
    >
      <div className="mb-1 flex justify-between text-text">
        <span>P(stenosis) per vessel</span>
        <span className="flex gap-2">
          {analysis && (
            <span className="text-muted">
              {mode === "f1" ? "max-F1" : "screening"}
            </span>
          )}
          <button
            type="button"
            className="text-accent"
            onClick={() => setUserOpen(false)}
            aria-expanded={true}
            aria-label="Collapse legend"
          >
            ▾
          </button>
        </span>
      </div>
      <div
        className="relative mt-1 h-3 rounded-sm"
        style={{ background: cssGradient() }}
        aria-hidden
      >
        {vessels.map(([id, r]) => (
          <div
            key={id}
            className="absolute -top-1 h-5 w-0.5 bg-text"
            style={{ left: `calc(${(r.threshold * 100).toFixed(1)}% - 1px)` }}
            title={`${id} threshold ${(r.threshold * 100).toFixed(0)}%`}
          />
        ))}
      </div>
      <div className="relative h-6">
        {vessels.map(([id, r], i) => (
          <span
            key={id}
            className="absolute -translate-x-1/2 font-mono text-[9px] text-text"
            style={{ left: `${r.threshold * 100}%`, top: i * 7 }}
          >
            {id}
          </span>
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px]">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
      {analysis && (
        <div className="mt-0.5">ticks: operating threshold per vessel</div>
      )}
      <div className="mt-1.5 grid grid-cols-2 gap-x-2 gap-y-1">
        <span className="flex items-center gap-1">
          <i
            className="inline-block h-2.5 w-5 rounded-sm"
            style={{ background: STRIPES }}
          />{" "}
          uncertain (conformal)
        </span>
        <span className="flex items-center gap-1">
          <i
            className="inline-block h-2.5 w-5 rounded-sm"
            style={{ background: NOT_MODELLED }}
          />{" "}
          not modelled
        </span>
        <span className="flex items-center gap-1">
          <i
            className="inline-block h-2.5 w-5 rounded-sm"
            style={{ background: NEUTRAL_VESSEL }}
          />{" "}
          no analysis
        </span>
      </div>
      {showTerritories && (
        <div
          className="mt-1.5 border-t border-line pt-1"
          data-testid="legend-territories"
        >
          <div>Heart wall: artery that usually supplies it (not risk)</div>
          <div className="mt-0.5 flex gap-2">
            {Object.entries(ARTERY_COLORS).map(([v, c]) => (
              <span key={v} className="flex items-center gap-1">
                <i
                  className="inline-block h-2.5 w-3 rounded-sm"
                  style={{ background: c }}
                />
                {v}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
