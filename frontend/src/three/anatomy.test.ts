import { describe, expect, it } from 'vitest'
import { bundledConfig } from '../api/client'
import { PROCEDURAL_NODES } from './procedural'
import manifestRaw from '../../public/models/anatomy_manifest.json?raw'

const cfg = bundledConfig()
const manifest = JSON.parse(manifestRaw) as {
  nodes: Record<string, { triangles: number }>
  total_triangles: number
  glb_bytes: number
  missing: string[]
}

describe('anatomy.yaml <-> targets.yaml', () => {
  const structures = cfg.anatomy.structures
  const vesselTargets = cfg.targets.vessels.map((v) => v.id)

  it('every vessel target maps to exactly one role:vessel structure', () => {
    for (const t of vesselTargets) {
      expect(structures.filter((s) => s.role === 'vessel' && s.target === t), t).toHaveLength(1)
    }
  })

  it('every vessel/territory structure target exists in targets.yaml', () => {
    for (const s of structures.filter((x) => x.role === 'vessel' || x.role === 'territory')) {
      expect(vesselTargets, `${s.node} -> ${s.target}`).toContain(s.target)
    }
  })

  it('cad is not mapped to geometry and node names are unique', () => {
    expect(structures.some((s) => s.target === cfg.targets.cad.id)).toBe(false)
    const names = structures.map((s) => s.node)
    expect(new Set(names).size).toBe(names.length)
  })

  it('bullseye vessels are targets', () => {
    for (const v of Object.keys(cfg.anatomy.bullseye)) expect(vesselTargets).toContain(v)
  })
})

describe('anatomy_manifest.json (built glb)', () => {
  it('contains every anatomy.yaml node with geometry', () => {
    for (const s of cfg.anatomy.structures) {
      expect(manifest.nodes[s.node], s.node).toBeDefined()
      expect(manifest.nodes[s.node].triangles, s.node).toBeGreaterThan(0)
    }
  })

  it('respects the budget and has no missing FMA IDs', () => {
    expect(manifest.total_triangles).toBeLessThanOrEqual(100_000)
    expect(manifest.glb_bytes).toBeLessThanOrEqual(3_000_000)
    expect(manifest.missing).toEqual([])
  })

  it('procedural fallback can draw every node', () => {
    for (const s of cfg.anatomy.structures) expect(PROCEDURAL_NODES, s.node).toContain(s.node)
  })
})
