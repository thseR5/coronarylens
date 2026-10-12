// 3D-only harness for e2e checks: the scene, legend and bullseye without the dashboard.
import { StrictMode, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import '../src/index.css'
import { api } from '../src/api/client'
import { useStore } from '../src/store'
import CoronaryScene from '../src/three/Scene'
import Bullseye from '../src/three/Bullseye'

function Harness() {
  const config = useStore((s) => s.config)
  const setConfig = useStore((s) => s.setConfig)
  useEffect(() => {
    api.config().then(setConfig)
  }, [setConfig])
  return (
    <div className="flex h-full">
      <section className="relative min-w-0 flex-1">{config && <CoronaryScene />}</section>
      <aside className="w-[280px] border-l border-line bg-panel p-3">{config && <Bullseye size={240} />}</aside>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Harness />
  </StrictMode>,
)
