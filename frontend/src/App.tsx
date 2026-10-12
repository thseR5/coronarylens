import { useEffect } from 'react'
import CoronaryScene from './three/Scene'
import { api } from './api/client'
import Dashboard from './dashboard/Dashboard'
import PatientInput from './dashboard/PatientInput'
import { useLiveAnalysis } from './dashboard/useLiveAnalysis'
import { DISCLAIMER } from './lib/disclaimer'
import { useStore } from './store'

export default function App() {
  const setConfig = useStore((s) => s.setConfig)
  const config = useStore((s) => s.config)
  const latencyMs = useLiveAnalysis()

  useEffect(() => {
    api.config().then(setConfig)
  }, [setConfig])

  return (
    <div className="flex h-full flex-col">
      <div role="alert" className="shrink-0 border-b border-warn/40 bg-warn/10 px-4 py-1.5 text-center text-xs text-warn sm:text-sm">
        ⚠ {DISCLAIMER}
      </div>
      <header className="flex shrink-0 items-center gap-3 border-b border-line px-4 py-2">
        <span className="font-semibold tracking-wide">CoronaryLens</span>
        <span className="hidden text-xs text-muted sm:inline">coherent, uncertainty-aware coronary risk twin</span>
      </header>
      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto xl:flex-row xl:overflow-hidden">
        <aside aria-label="Patient inputs" className="order-2 border-line bg-panel p-3 xl:order-1 xl:w-[330px] xl:shrink-0 xl:overflow-y-auto xl:border-r">
          <h2 className="h-section">Patient</h2>
          {config && <PatientInput />}
        </aside>
        <section aria-label="3D heart" className="relative order-1 h-[55vh] shrink-0 xl:order-2 xl:h-auto xl:min-w-0 xl:flex-1">
          {config && <CoronaryScene />}
        </section>
        <aside aria-label="Risk dashboard" className="order-3 border-line bg-panel xl:w-[460px] xl:shrink-0 xl:overflow-y-auto xl:border-l">
          {config && <Dashboard latencyMs={latencyMs} />}
        </aside>
      </main>
      <footer className="shrink-0 border-t border-line px-4 py-1.5 text-[11px] text-muted">
        {DISCLAIMER} · Data: Z-Alizadeh Sani (UCI, CC BY 4.0) · Anatomy: BodyParts3D © DBCLS (CC BY 4.0)
      </footer>
    </div>
  )
}
