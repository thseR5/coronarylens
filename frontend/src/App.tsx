import { useEffect } from 'react'
import CoronaryScene from './three/Scene'
import { api } from './api/client'
import { DISCLAIMER } from './lib/disclaimer'
import { useStore } from './store'

export default function App() {
  const setConfig = useStore((s) => s.setConfig)
  const config = useStore((s) => s.config)

  useEffect(() => {
    api.config().then(setConfig)
  }, [setConfig])

  return (
    <div className="flex h-full flex-col">
      <div role="alert" className="border-b border-warn/40 bg-warn/10 px-4 py-2 text-center text-sm text-warn">
        ⚠ {DISCLAIMER}
      </div>
      <header className="flex items-center gap-3 border-b border-line px-4 py-2">
        <span className="font-semibold tracking-wide">CoronaryLens</span>
        <span className="text-xs text-muted">coherent, uncertainty-aware coronary risk twin</span>
      </header>
      <main className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <section className="relative min-h-[50vh] flex-1 lg:min-h-0">{config && <CoronaryScene />}</section>
        <aside className="w-full overflow-y-auto border-line bg-panel lg:w-[440px] lg:border-l">
          <div className="p-4 text-sm text-muted">Dashboard</div>
        </aside>
      </main>
      <footer className="border-t border-line px-4 py-1.5 text-[11px] text-muted">
        {DISCLAIMER} · Data: Z-Alizadeh Sani (UCI, CC BY 4.0) · Anatomy: BodyParts3D © DBCLS (CC BY 4.0)
      </footer>
    </div>
  )
}
