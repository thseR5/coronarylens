import { useState } from 'react'
import { mockAnalysis, mockExplanation } from '../api/mock'
import { useStore } from '../store'
import About from './About'
import MeasurementsTable from './MeasurementsTable'
import RiskSummary from './RiskSummary'
import SimilarPatients from './SimilarPatients'
import VesselCards from './VesselCards'
import WhatIf from './WhatIf'
import WhyPanel from './WhyPanel'
import ModelPerformance from './ModelPerformance'
import Bullseye from '../three/Bullseye'

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
        {latencyMs != null && <span className="ml-auto self-center font-mono text-[10px] text-muted">{latencyMs} ms</span>}
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
          </div>
          <WhyPanel />
          <MeasurementsTable />
          <WhatIf />
          <SimilarPatients />
        </>
      )}
      {tab === 'model' && <ModelPerformance />}
      {tab === 'about' && <About />}
    </div>
  )
}
