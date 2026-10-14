import { useState } from 'react'
import { mockAnalysis, mockExplanation } from '../api/mock'
import { useStore } from '../store'
import About from './About'
import MeasurementsTable from './MeasurementsTable'
import RiskSummary from './RiskSummary'
import SimilarPatients from './SimilarPatients'
import CohortMap from './CohortMap'
import { api } from '../api/client'
import VesselCards from './VesselCards'
import WhatIf from './WhatIf'
import WhyPanel from './WhyPanel'
import ModelPerformance from './ModelPerformance'
import Bullseye from '../three/Bullseye'
import TerritoryAgreement from './TerritoryAgreement'

type Tab = 'analysis' | 'model' | 'about'

export default function Dashboard({ latencyMs }: { latencyMs: number | null }) {
  const [tab, setTab] = useState<Tab>('analysis')
  const error = useStore((s) => s.error)
  const analysis = useStore((s) => s.analysis)

  return (
    <div className="flex flex-col gap-3 p-3">
      <div role="tablist" aria-label="Dashboard sections" className="flex gap-1 border-b border-line">
        {(
          [
            ['analysis', 'Patient analysis'],
            ['model', 'Model performance'],
            ['about', 'About & limits'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            role="tab"
            type="button"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={`-mb-px border-b-2 px-2 py-1.5 text-xs ${tab === id ? 'border-accent text-text' : 'border-transparent text-muted hover:text-text'}`}
          >
            {label}
          </button>
        ))}
        <span className="ml-auto flex items-center gap-2 self-center">
          {latencyMs != null && <span className="font-mono text-[10px] text-muted">{latencyMs} ms</span>}
          <button
            type="button"
            className="btn !py-0.5"
            disabled={!analysis || analysis.model_version === 'mock'}
            title="Printable report with the disclaimer; use the browser's Print dialog to save it as PDF"
            onClick={() => {
              // open the report by URL (not by writing into a blank popup): works inside sandboxed frames such as
              // a Hugging Face Space; the report page opens the print dialog itself
              window.open(api.reportUrl(useStore.getState().features, true), '_blank')
            }}
          >
            Report / PDF
          </button>
        </span>
      </div>

      {error && !analysis && (
        <div role="alert" className="flex items-center justify-between gap-2 rounded-md bg-warn/10 p-2 text-xs text-warn ring-1 ring-warn/40">
          <span>Model backend unreachable. Start it with <code>make dev-api</code>.</span>
          <button
            type="button"
            className="btn"
            onClick={() => {
              useStore.getState().setAnalysis(mockAnalysis)
              useStore.getState().setExplanation(mockExplanation)
            }}
          >
            Show demo data
          </button>
        </div>
      )}
      {analysis?.model_version === 'mock' && (
        <div className="rounded-md bg-warn/10 px-2 py-1 text-center text-[11px] text-warn">DEMO DATA: not a model prediction</div>
      )}

      {tab === 'analysis' && (
        <>
          <RiskSummary />
          <VesselCards />
          <div className="card flex flex-col items-center gap-1">
            <h2 className="h-section self-start">AHA 17-segment map · expected supply territory</h2>
            <Bullseye size={200} />
            <TerritoryAgreement />
          </div>
          <WhyPanel />
          <MeasurementsTable />
          <WhatIf />
          <SimilarPatients />
          <CohortMap />
        </>
      )}
      {tab === 'model' && <ModelPerformance />}
      {tab === 'about' && <About />}
    </div>
  )
}
