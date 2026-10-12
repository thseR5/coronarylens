import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { bundledConfig } from '../api/client'
import { mockAnalysis, mockExplanation } from '../api/mock'
import { useStore } from '../store'
import Dashboard from './Dashboard'
import { defaultFeatures } from './PatientInput'

describe('Dashboard', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('offline'))))
    const config = bundledConfig()
    useStore.setState({
      config,
      features: defaultFeatures(config.features.features),
      analysis: mockAnalysis,
      explanation: mockExplanation,
      selected: null,
    })
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('shows CAD and one card per configured vessel', () => {
    render(<Dashboard latencyMs={42} />)
    expect(screen.getAllByText('92%').length).toBeGreaterThan(0)
    for (const v of bundledConfig().targets.vessels) expect(screen.getByRole('button', { name: new RegExp(`^${v.id}`), pressed: false })).toBeTruthy()
  })

  it('switches the explanation target when a vessel card is clicked', () => {
    render(<Dashboard latencyMs={null} />)
    fireEvent.click(screen.getByRole('button', { name: /^LCX/ }))
    expect(useStore.getState().selected).toBe('LCX')
    expect(screen.getByRole('tab', { name: 'LCX', selected: true })).toBeTruthy()
  })

  it('every input feature in the config appears in the measurements table', () => {
    render(<Dashboard latencyMs={null} />)
    for (const f of bundledConfig().features.features) expect(screen.getAllByText(f.label).length).toBeGreaterThan(0)
  })
})
