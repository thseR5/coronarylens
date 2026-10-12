import type { ReactNode } from 'react'
import { useStore, type ViewState } from '../store'
import { useSceneLocal } from './sceneState'

function Btn({ active, onClick, children, title, testId }: {
  active?: boolean
  onClick: () => void
  children: ReactNode
  title?: string
  testId?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-pressed={active}
      data-testid={testId}
      className={`rounded border px-2 py-0.5 text-[11px] focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
        active ? 'border-accent bg-accent/15 text-text' : 'border-line bg-panel/80 text-muted hover:text-text'
      }`}
    >
      {children}
    </button>
  )
}

/** Overlay toolbar: angiographic presets, layer toggles, x-ray, auto-rotate, heartbeat, labels, PNG export. */
export default function SceneControls() {
  const presets = useStore((s) => s.config?.anatomy.camera_presets)
  const voronoiLabel = useStore((s) => s.config?.anatomy.voronoi_label)
  const view = useStore((s) => s.view)
  const setView = useStore((s) => s.setView)
  const local = useSceneLocal()
  const toggle = (k: keyof Omit<ViewState, 'cameraPreset' | 'territoryMode'>) => setView({ [k]: !view[k] })

  const screenshot = () => {
    const url = local.screenshot?.()
    if (!url) return
    const a = document.createElement('a')
    a.href = url
    a.download = `coronarylens-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`
    a.click()
  }

  return (
    <div className="pointer-events-auto flex max-w-full flex-col gap-1" role="toolbar" aria-label="3D view controls">
      <div className="flex flex-wrap gap-1" aria-label="Camera presets">
        {(presets ?? []).map((p) => (
          <Btn key={p.id} onClick={() => setView({ cameraPreset: p.id })} testId={`preset-${p.id}`} title={`Camera: ${p.label}`}>
            {p.label}
          </Btn>
        ))}
      </div>
      <div className="flex flex-wrap gap-1" aria-label="Layers">
        <Btn active={view.showRibs} onClick={() => toggle('showRibs')}>Ribs</Btn>
        <Btn active={view.showLungs} onClick={() => toggle('showLungs')} title="Bronchial tree and pulmonary vessels (BodyParts3D 4.0 has no lung-surface mesh)">Airways &amp; pulmonary vessels</Btn>
        <Btn active={view.showAorta} onClick={() => toggle('showAorta')}>Aorta</Btn>
        <Btn active={view.showTerritories} onClick={() => toggle('showTerritories')} testId="toggle-territories">Territories</Btn>
        {view.showTerritories && (
          <Btn
            active={view.territoryMode === 'voronoi'}
            onClick={() => setView({ territoryMode: view.territoryMode === 'aha' ? 'voronoi' : 'aha' })}
            testId="toggle-territory-mode"
            title="Default: the 17 AHA segments with the standard vessel assignment. Toggle: nearest-artery (Voronoi) partition of the LV surface."
          >
            {view.territoryMode === 'aha' ? 'AHA 17-segment' : (voronoiLabel ?? 'nearest-artery estimate')}
          </Btn>
        )}
        <Btn active={view.xray} onClick={() => toggle('xray')} testId="toggle-xray">X-ray</Btn>
        <Btn active={view.autoRotate} onClick={() => toggle('autoRotate')} testId="toggle-rotate">Rotate</Btn>
        <Btn active={local.heartbeat} onClick={() => local.set({ heartbeat: !local.heartbeat })} testId="toggle-heartbeat">Heartbeat</Btn>
        <Btn active={local.labels} onClick={() => local.set({ labels: !local.labels })}>Labels</Btn>
        <Btn onClick={screenshot} title="Save the 3D view as PNG (labels not included)" testId="screenshot">PNG</Btn>
      </div>
    </div>
  )
}
